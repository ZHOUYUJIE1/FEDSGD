import os
from typing import Dict, Tuple, Optional

import torch
from torch import nn
from torch.autograd import Variable

from model.lenet import lenet5, resnet18
from model.logger import get_logger

# Your existing top-k compressor (supports adaptive/fixed/none + optional encryption/quantization)
# from compare.cate_compressor import CATECompressor

# New baseline from FL-CATE paper (dynamic sparsification + error compensation)
try:
    from compare.cate_compressor import CATECompressor
except Exception:
    CATECompressor = None


class client(object):
    """
    Client-side gradient computation + (optional) compression.
    Supported compression_mode:
      - 'none'    : no sparsification, return dense grads
      - 'fixed'   : Top-K with fixed keep ratio (TopkCompressor)
      - 'adaptive': your dynamic sparsification (TopkCompressor)
      - 'cate'    : FL-CATE baseline (dynamic B selection + error feedback), returns sparse-then-decompressed dense grads
    """

    def __init__(
        self,
        rank: int,
        data_loader: Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader],
        device: Optional[torch.device] = None,
        compress_ratio: float = 0.1,
        min_sparsity_ratio: float = 0.1,
        max_sparsity_ratio: float = 0.9,
        encryption_key: Optional[str] = None,
        n_class: int = 10,
        logger=None,
        compression_mode: str = "adaptive",
        in_dim: int = 3,
        model_type: str = "lenet5",
        # CATE-specific (baseline) knobs
        cate_value_bits: int = 32,
        cate_packet_overhead_bits: int = 0,
        cate_index_overhead_bits: int = 0,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        seed = 19201077 + 19950920 + rank
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        self.rank = rank
        self.n_class = n_class
        self.in_dim = in_dim
        self.model_type = model_type
        self.logger = logger

        self.train_loader = data_loader[0]
        self.test_loader = data_loader[1]

        self.compression_mode = compression_mode

        # Fixed batch cache (DLG-friendly)
        self._fixed_batch_cpu = None

        # Instantiate compressor depending on mode
        if self.compression_mode == "cate":
            if CATECompressor is None:
                raise ImportError("compression_mode='cate' requires cate_compressor.py in PYTHONPATH.")
            self.compressor = CATECompressor(
                value_bits=cate_value_bits,
                packet_overhead_bits=cate_packet_overhead_bits,
                index_overhead_bits=cate_index_overhead_bits,
                cost_model="bits",
                logger=logger,
            )
        else:
            self.compressor = TopkCompressor(
                compress_ratio=compress_ratio,
                min_sparsity_ratio=min_sparsity_ratio,
                max_sparsity_ratio=max_sparsity_ratio,
                encryption_key=encryption_key,
                logger=logger,
                compression_mode=compression_mode,
            )

    def __model_fn(self):
        return resnet18 if self.model_type == "resnet18" else lenet5

    def __load_global_model(self):
        model_fn = self.__model_fn()

        if os.path.exists("./cache/global_model_state.pkl"):
            global_model_state = torch.load("./cache/global_model_state.pkl", map_location=self.device)
            model = model_fn(n_class=self.n_class, in_dim=self.in_dim).to(self.device)
            try:
                model.load_state_dict(global_model_state)
            except RuntimeError:
                # if self.logger:
                #     self.logger.warning(
                #         f"[Client {self.rank}] global_model_state mismatch; rebuilding model with n_class={self.n_class}, in_dim={self.in_dim}"
                #     )
                print(f"[Client {self.rank}] global_model_state mismatch; rebuilding model with n_class={self.n_class}, in_dim={self.in_dim}")
                model = model_fn(n_class=self.n_class, in_dim=self.in_dim).to(self.device)
        else:
            model = model_fn(n_class=self.n_class, in_dim=self.in_dim).to(self.device)

        normalized_direction = None
        if os.path.exists("./cache/normalized_direction.pkl"):
            normalized_direction = torch.load("./cache/normalized_direction.pkl", map_location=self.device)
            if normalized_direction is not None:
                normalized_direction = {
                    k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in normalized_direction.items()
                }
        return model, normalized_direction

    def __compute_grads_fixed_batch(self, model: torch.nn.Module):
        """Compute grads on a cached fixed batch (one backward)."""
        criterion = nn.CrossEntropyLoss()
        model.train()

        if self._fixed_batch_cpu is None:
            data0, target0 = next(iter(self.train_loader))
            self._fixed_batch_cpu = (data0.detach().cpu().clone(), target0.detach().cpu().clone())
            if self.logger:
                self.logger.info(f"[Client {self.rank}] Cached fixed batch: batch_size={int(data0.size(0))}")

        data_cpu, target_cpu = self._fixed_batch_cpu
        data = Variable(data_cpu).to(self.device)
        target = Variable(target_cpu).to(self.device)

        output = model(data)
        loss = criterion(output, target)

        batch_size = int(data.size(0))
        train_loss = float(loss.item()) * batch_size
        pred = output.argmax(dim=1, keepdim=True)
        train_correct = int(pred.eq(target.view_as(pred)).sum().item())

        model.zero_grad(set_to_none=True)
        loss.backward()

        raw_grads = {}
        for name, param in model.named_parameters():
            if param.grad is None:
                raw_grads[name] = torch.zeros_like(param.data)
            else:
                g = param.grad.detach()
                if torch.isnan(g).any() or torch.isinf(g).any():
                    if self.logger:
                        self.logger.warning(f"[Client {self.rank}] NaN/Inf grad in {name}; replacing with 0")
                    g = torch.where(torch.isnan(g) | torch.isinf(g), torch.zeros_like(g), g)
                raw_grads[name] = g.clone()

        # Weight for aggregation: batch_size (since this is single-batch FedSGD)
        n_samples_for_weight = batch_size
        acc_denom = batch_size
        return raw_grads, train_loss, train_correct, acc_denom, n_samples_for_weight

    def __compute_grads_full(self, model: torch.nn.Module):
        """Original FedSGD-like logic: iterate local data, average grads."""
        criterion = nn.CrossEntropyLoss()
        model.train()

        train_loss = 0.0
        train_correct = 0

        num_local_epochs = 1 if self.model_type == "lenet5" else 3

        accumulated_grads = {name: torch.zeros_like(param) for name, param in model.named_parameters()}

        for _ in range(num_local_epochs):
            epoch_grad_sums = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
            epoch_samples = 0

            for data, target in self.train_loader:
                data = Variable(data).to(self.device)
                target = Variable(target).to(self.device)
                batch_size = int(data.size(0))
                epoch_samples += batch_size

                output = model(data)
                loss = criterion(output, target)

                train_loss += float(loss.item()) * batch_size
                pred = output.argmax(dim=1, keepdim=True)
                train_correct += int(pred.eq(target.view_as(pred)).sum().item())

                model.zero_grad(set_to_none=True)
                loss.backward()

                for name, param in model.named_parameters():
                    if param.grad is None:
                        continue
                    g = param.grad.detach()
                    if torch.isnan(g).any() or torch.isinf(g).any():
                        if self.logger:
                            self.logger.warning(f"[Client {self.rank}] NaN/Inf grad in {name}; replacing with 0")
                        g = torch.where(torch.isnan(g) | torch.isinf(g), torch.zeros_like(g), g)
                    epoch_grad_sums[name] += g * batch_size

            denom = max(epoch_samples, 1)
            for name in epoch_grad_sums:
                accumulated_grads[name] += epoch_grad_sums[name] / denom

        raw_grads = {name: grad / num_local_epochs for name, grad in accumulated_grads.items()}
        acc_denom = len(self.train_loader.dataset)
        n_samples_for_weight = len(self.train_loader.dataset)
        return raw_grads, train_loss, train_correct, acc_denom, n_samples_for_weight

    def __compress_grads(self, raw_grads: Dict[str, torch.Tensor], normalized_direction: Optional[Dict[str, torch.Tensor]], enable_compression: bool, enable_compression_all: bool):
        """
        Return (named_grads, sparsity_ratio, encrypted_flag, effective_mode)
        """
        round_enabled = bool(enable_compression and enable_compression_all)
        if not round_enabled or self.compression_mode == "none":
            return raw_grads, 0.0, False, "none"

        # ---- FL-CATE baseline ----
        if self.compression_mode == "cate":
            sparse_grads = {}
            total_params = 0
            sparse_params = 0
            for name, local_grad in raw_grads.items():
                # CATE selects B internally based on (Δ+e)
                values, indices = self.compressor.compress_tensor(local_grad, name=name)
                dense_sparse = self.compressor.decompress_tensor(values, indices, local_grad.shape)
                sparse_grads[name] = dense_sparse
                total_params += local_grad.numel()
                sparse_params += int((dense_sparse != 0).sum().item())

            sparsity_ratio = 1.0 - (sparse_params / total_params) if total_params > 0 else 0.0
            # No encryption/quantization in this baseline implementation
            return sparse_grads, sparsity_ratio, False, "cate"

        # ---- Your TopK pipeline (adaptive/fixed) ----
        sparse_grads = {}
        for name, local_grad in raw_grads.items():
            if normalized_direction is not None and name in normalized_direction:
                global_grad = normalized_direction[name]
                values, indices = self.compressor.compress_tensor(local_grad, global_gradient=global_grad)
            else:
                values, indices = self.compressor.compress_tensor(local_grad)
            sparse_grads[name] = self.compressor.decompress_tensor(values, indices, local_grad.shape)

        # encryption + quantization stage (your existing method)
        encrypted_grads = self.compressor.compress_with_encryption(sparse_grads)

        total_params = sum(g.numel() for g in raw_grads.values())
        sparse_params = sum(int((g != 0).sum().item()) for g in sparse_grads.values())
        sparsity_ratio = 1.0 - (sparse_params / total_params) if total_params > 0 else 0.0

        return encrypted_grads, sparsity_ratio, True, self.compression_mode

    def __train(self, model, normalized_direction, enable_compression, enable_compression_all, use_fixed_batch: bool = False):
        # Compute raw grads
        if use_fixed_batch:
            raw_grads, train_loss, train_correct, acc_denom, n_samples_for_weight = self.__compute_grads_fixed_batch(model)
        else:
            raw_grads, train_loss, train_correct, acc_denom, n_samples_for_weight = self.__compute_grads_full(model)

        named_grads, sparsity_ratio, encrypted_flag, effective_mode = self.__compress_grads(
            raw_grads, normalized_direction, enable_compression, enable_compression_all
        )

        # if self.logger:
        #     self.logger.info(
        #         f"[Rank {self.rank}] Loss: {train_loss:.6f}, Acc: {(train_correct / max(acc_denom,1)):.4f}, "
        #         f"Mode: {effective_mode}, Sparsity: {sparsity_ratio*100.0:.2f}%, Encrypted: {encrypted_flag}"
        #     )
        print(f"[Rank {self.rank}] Loss: {train_loss:.6f}, Acc: {(train_correct / max(acc_denom,1)):.4f}, "
                f"Mode: {effective_mode}, Sparsity: {sparsity_ratio*100.0:.2f}%, Encrypted: {encrypted_flag}")

        grads = {
            "n_samples": n_samples_for_weight,
            "named_grads": named_grads,
            "compression_mode": effective_mode,
            "encrypted": bool(encrypted_flag),
            "enable_compression_all": bool(enable_compression_all),
            "enable_compression": bool(enable_compression),
            "fixed_batch": bool(use_fixed_batch),
        }
        return grads

    def run(self, enable_compression: bool, enable_compression_all: bool, use_fixed_batch: bool = False):
        model, normalized_direction = self.__load_global_model()
        grads = self.__train(
            model=model,
            normalized_direction=normalized_direction,
            enable_compression=enable_compression,
            enable_compression_all=enable_compression_all,
            use_fixed_batch=use_fixed_batch,
        )
        os.makedirs("./cache", exist_ok=True)
        torch.save(grads, f"./cache/grads_{self.rank}.pkl")
