import glob
import logging
import os
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.autograd import Variable

from model.lenet import lenet5, resnet18

# For decrypting grads when using your TopkCompressor pipeline (adaptive/fixed with encryption)
try:
    from topk import TopkCompressor
except Exception:
    TopkCompressor = None


class server(object):
    """
    Center server:
      - reads aggregated gradients from intermediate servers (server1) if present;
        otherwise can fall back to reading client grads directly.
      - supports compression_mode:
          'none'/'fixed'/'adaptive' : same as your original pipeline (may include encrypted payloads)
          'cate'                   : FL-CATE baseline produces dense tensors (zero-filled) with encrypted=False
    """

    def __init__(
        self,
        size: int,
        data_loader: Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader],
        device: Optional[torch.device] = None,
        encryption_key: Optional[str] = None,
        n_class: int = 10,
        logger=None,
        enable_attack: bool = False,
        attack_types: Optional[List[str]] = None,
        compression_mode: str = "adaptive",
        in_dim: int = 3,
        model_type: str = "lenet5",
        lr_lenet: float = 0.01,
        lr_resnet: float = 0.1,
        momentum: float = 0.9,
        weight_decay_resnet: float = 5e-4,
    ):
        self.size = int(size)
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encryption_key = encryption_key
        self.n_class = int(n_class)
        self.logger = logger
        self.enable_attack = bool(enable_attack)
        self.attack_types = attack_types or []
        self.compression_mode = compression_mode
        self.in_dim = int(in_dim)
        self.model_type = model_type

        self.train_loader = data_loader[0]
        self.test_loader = data_loader[1]

        # Model
        model_fn = resnet18 if self.model_type == "resnet18" else lenet5
        self.model = model_fn(n_class=self.n_class, in_dim=self.in_dim).to(self.device)

        # Load global state if exists
        if os.path.exists("./cache/global_model_state.pkl"):
            try:
                state = torch.load("./cache/global_model_state.pkl", map_location=self.device)
                self.model.load_state_dict(state)
                self._log_info("Loaded existing global model state from ./cache/global_model_state.pkl")
            except Exception as e:
                self._log_warn(f"Failed to load global model state; starting fresh. err={e}")

        # Optimizer (FedSGD-style, server applies aggregated grads)
        if self.model_type == "resnet18":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr_resnet, momentum=momentum, weight_decay=weight_decay_resnet)
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr_lenet, momentum=momentum)

        # Decryptor (only needed when your TopkCompressor encryption is used)
        self._decryptor = None
        if TopkCompressor is not None and self.encryption_key is not None:
            try:
                self._decryptor = TopkCompressor(
                    compress_ratio=1.0,
                    min_sparsity_ratio=0.0,
                    max_sparsity_ratio=1.0,
                    encryption_key=self.encryption_key,
                    logger=self.logger,
                    compression_mode="none",
                )
            except Exception as e:
                self._log_warn(f"TopkCompressor init failed; encrypted grads may not decrypt. err={e}")

    # ---------------- utils ----------------
    def _log_info(self, msg: str):
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

    def _log_warn(self, msg: str):
        if self.logger:
            self.logger.warning(msg)
        else:
            print("WARNING:", msg)

    # ---------------- IO: find incoming grads ----------------
    def _find_server1_grad_files(self) -> List[str]:
        """
        Try common filename patterns produced by different server1 implementations.
        We prefer server1 outputs; if none, caller may fall back to client grads.
        """
        patterns = [
            "./cache/grads_server1_*.pkl",
            "./cache/server1_grads_*.pkl",
            "./cache/grads_s1_*.pkl",
            "./cache/grads_intermediate_*.pkl",
        ]
        files = []
        for pat in patterns:
            files.extend(glob.glob(pat))
        # de-dup
        files = sorted(list(dict.fromkeys(files)))
        return files

    def _find_client_grad_files(self) -> List[str]:
        return sorted(glob.glob("./cache/grads_*.pkl"))

    # ---------------- core: aggregate ----------------
    def aggregate(self, epoch: int = 0):
        """
        1) Load server1 aggregated grads if available, else load client grads directly.
        2) Decrypt if needed.
        3) Weighted average by n_samples.
        4) Apply to model using optimizer.step().
        5) Save global model state.
        6) Evaluate.
        """
        os.makedirs("./cache", exist_ok=True)

        grad_files = self._find_server1_grad_files()
        source = "server1"
        if len(grad_files) == 0:
            grad_files = self._find_client_grad_files()
            source = "client"
        if len(grad_files) == 0:
            self._log_warn("No gradient files found in ./cache. Skip aggregation.")
            return

        self._log_info(f"[Center] Round {epoch}: aggregating from {source} files={len(grad_files)}")

        payloads = []
        for fp in grad_files:
            try:
                payloads.append(torch.load(fp, map_location=self.device))
            except Exception as e:
                self._log_warn(f"Failed to load {fp}: {e}")

        if len(payloads) == 0:
            self._log_warn("All gradient files failed to load. Skip aggregation.")
            return

        # Decrypt if needed (TopkCompressor pipeline sets encrypted=True)
        for p in payloads:
            if not isinstance(p, dict):
                continue
            enc = bool(p.get("encrypted", False))
            if enc:
                if self._decryptor is None:
                    self._log_warn("Encrypted gradients present but decryptor not available; using as-is.")
                    continue
                try:
                    # Your project typically provides decrypt_with_encryption; try common method names.
                    named = p.get("named_grads", {})
                    if hasattr(self._decryptor, "decrypt_with_encryption"):
                        p["named_grads"] = self._decryptor.decrypt_with_encryption(named)
                    elif hasattr(self._decryptor, "decompress_with_encryption"):
                        p["named_grads"] = self._decryptor.decompress_with_encryption(named)
                    else:
                        self._log_warn("Decryptor has no decrypt_with_encryption/decompress_with_encryption; using as-is.")
                except Exception as e:
                    self._log_warn(f"Decrypt failed: {e}")

        # Weighted average
        agg: Dict[str, torch.Tensor] = {}
        total_weight = 0.0

        for p in payloads:
            if not isinstance(p, dict):
                continue
            named: Dict[str, torch.Tensor] = p.get("named_grads", {})
            w = float(p.get("n_samples", 1.0))
            if w <= 0:
                w = 1.0
            total_weight += w

            for name, g in named.items():
                if g is None:
                    continue
                gt = g.to(self.device)
                if name not in agg:
                    agg[name] = gt * w
                else:
                    agg[name] += gt * w

        if total_weight <= 0:
            total_weight = 1.0
        for name in list(agg.keys()):
            agg[name] = agg[name] / total_weight

        # Apply update
        self.__step(agg)

        # Save global model
        torch.save(self.model.state_dict(), "./cache/global_model_state.pkl")
        self._log_info("[Center] Saved global model to ./cache/global_model_state.pkl")

        # Evaluate
        self.__evaluate(epoch)

    # ---------------- apply gradients ----------------
    def __step(self, gradients: Dict[str, torch.Tensor]):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        # Assign aggregated grads to parameters
        name_to_param = {n: p for n, p in self.model.named_parameters()}
        for name, g in gradients.items():
            if name not in name_to_param:
                continue
            p = name_to_param[name]
            if g is None:
                continue
            gg = g.detach()
            if torch.isnan(gg).any() or torch.isinf(gg).any():
                self._log_warn(f"[Center] NaN/Inf in aggregated grad {name}; replacing with 0")
                gg = torch.where(torch.isnan(gg) | torch.isinf(gg), torch.zeros_like(gg), gg)
            p.grad = gg

        self.optimizer.step()

    # ---------------- evaluation ----------------
    @torch.no_grad()
    def __evaluate(self, epoch: int):
        self.model.eval()
        criterion = nn.CrossEntropyLoss()

        test_loss = 0.0
        correct = 0
        total = 0

        for data, target in self.test_loader:
            data = data.to(self.device)
            target = target.to(self.device)
            out = self.model(data)
            loss = criterion(out, target)

            bs = int(data.size(0))
            test_loss += float(loss.item()) * bs
            pred = out.argmax(dim=1)
            correct += int((pred == target).sum().item())
            total += bs

        if total <= 0:
            total = 1

        self._log_info(
            f"[Center] Round {epoch}: TestLoss={test_loss/total:.6f}, TestAcc={correct/total:.4f} (n={total})"
        )
