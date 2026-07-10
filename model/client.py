import os
import torch

from torch import nn
from torch.autograd import Variable

from model.lenet import lenet5, resnet18, cifar10cnn_v2
from topk import TopkCompressor
from model.logger import get_logger


class client(object):
    def __init__(self, rank, data_loader, device=None, compress_ratio=0.1, min_sparsity_ratio=0.1, max_sparsity_ratio=0.9, encryption_key=None, n_class=10, logger=None, compression_mode='adaptive', in_dim=3, model_type='lenet5',unlearn_mode='unlearn'):
        """
        Args:
            rank: 客户端编号
            data_loader: 数据加载器
            device: 设备
            compress_ratio: 压缩率（用于fixed模式）
            min_sparsity_ratio: 最小稀疏率
            max_sparsity_ratio: 最大稀疏率
            encryption_key: 加密密钥
            n_class: 类别数，默认10（CIFAR10/MNIST），CIFAR100需要设置为100
            logger: 日志记录器
            compression_mode: 压缩模式
                - 'adaptive': 动态稀疏化压缩（根据global_gradient计算adaptive_sparsity）
                - 'fixed': 固定稀疏度压缩（使用compress_ratio）
                - 'none': 不使用稀疏压缩（返回完整梯度）
            in_dim: 输入通道数，默认3（CIFAR），MNIST需要设置为1
            model_type: 模型类型，'lenet5'、'resnet18' 或 'cifar10cnn_v2'，默认 'lenet5'
        """
        # 设备配置：自动检测GPU是否可用
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # seed
        seed = 19201077 + 19950920 + rank
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # rank
        self.rank = rank
        self.n_class = n_class  # 保存类别数
        self.in_dim = in_dim  # 保存输入通道数
        self.model_type = model_type  # 保存模型类型

        # data loader
        self.train_loader = data_loader[0]
        self.test_loader = data_loader[1]
        self.logger = logger  # 保存日志器
        # 初始化TopK压缩器，用于梯度稀疏化（传入加密密钥）
        self.compressor = TopkCompressor(
            compress_ratio=compress_ratio,
            min_sparsity_ratio=min_sparsity_ratio,
            max_sparsity_ratio=max_sparsity_ratio,
            encryption_key=encryption_key,  # 传入加密密钥
            logger=logger,
            compression_mode=compression_mode,  # 传入压缩模式
            unlearn_mode=unlearn_mode
        )

        

    
        # 固定 batch 缓存（用于每轮复用同一 batch 做梯度，便于重建对比）
        self._fixed_batch_cpu = None
    def __load_global_model(self):
        # 根据模型类型选择模型创建函数
        if self.model_type == 'resnet18':
            model_fn = resnet18
        elif self.model_type == 'cifar10cnn_v2':
            model_fn = cifar10cnn_v2
        else:
            model_fn = lenet5
        
        # 检查模型文件是否存在
        if os.path.exists('./cache/global_model_state.pkl'):
            global_model_state = torch.load('./cache/global_model_state.pkl', map_location=self.device)
            # 使用传入的类别数和输入通道数创建模型
            model = model_fn(n_class=self.n_class, in_dim=self.in_dim).to(self.device)
            # 尝试加载模型状态，如果类别数或输入通道数不匹配会报错
            try:
                model.load_state_dict(global_model_state)
            except RuntimeError as e:
                # 如果类别数或输入通道数不匹配，需要重新创建模型
                print(f'[Client {self.rank}] Warning: 模型参数不匹配，使用新的类别数 {self.n_class} 和输入通道数 {self.in_dim} 重新创建模型')
                model = model_fn(n_class=self.n_class, in_dim=self.in_dim).to(self.device)
        else:
            # 如果模型文件不存在，创建新模型
            model = model_fn(n_class=self.n_class, in_dim=self.in_dim).to(self.device)
        
        # 加载归一化方向，如果文件不存在则返回None（第一次运行时）
        normalized_direction = None
        if os.path.exists('./cache/normalized_direction.pkl'):
            normalized_direction = torch.load('./cache/normalized_direction.pkl', map_location=self.device)
            # 确保所有张量都在正确的设备上
            if normalized_direction is not None:
                normalized_direction = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                                       for k, v in normalized_direction.items()}
        
        return model, normalized_direction
    def __train(self, model, normalized_direction, enable_compression, enable_compression_all, use_fixed_batch=False):
        """本地训练并生成（可选压缩/加密的）梯度包。

        两种模式：
        - use_fixed_batch=False（默认）：保留你原来的 FedSGD 逻辑（遍历本地数据、可多 local epochs，并做 per-sample 平均）
        - use_fixed_batch=True：每轮只用同一个固定 batch 做一次 backward，便于 DLG/IG 对比实验（重建一个 batch）
        """
        criterion = nn.CrossEntropyLoss()
        model.train()

        # ===== 固定 batch 模式：每轮只对同一 batch 做一次 backward（DLG-friendly）=====
        if use_fixed_batch:
            # 第一次调用时缓存一个 batch 到 CPU，后续每轮复用，保证完全一致
            if not hasattr(self, "_fixed_batch_cpu") or self._fixed_batch_cpu is None:
                data0, target0 = next(iter(self.train_loader))
                self._fixed_batch_cpu = (data0.detach().cpu().clone(), target0.detach().cpu().clone())
                if self.logger:
                    self.logger.info(f"[Client {self.rank}] Cached fixed batch: batch_size={int(data0.size(0))}")
                else:
                    print(f"[Client {self.rank}] Cached fixed batch: batch_size={int(data0.size(0))}")

            data_cpu, target_cpu = self._fixed_batch_cpu
            data = Variable(data_cpu).to(self.device)
            target = Variable(target_cpu).to(self.device)
            batch_size = int(data.size(0))

            output = model(data)
            loss = criterion(output, target)

            # 统计（仅该 batch）
            train_loss = float(loss.item()) * batch_size
            pred = output.argmax(dim=1, keepdim=True)
            train_correct = int(pred.eq(target.view_as(pred)).sum().item())
            acc_denom = batch_size

            model.zero_grad(set_to_none=True)
            loss.backward()

            raw_grads = {}
            for name, param in model.named_parameters():
                if param.grad is None:
                    raw_grads[name] = torch.zeros_like(param.data)
                    continue
                g = param.grad.detach()
                if torch.isnan(g).any() or torch.isinf(g).any():
                    if self.logger:
                        self.logger.warning(f"[Client {self.rank}] 梯度包含NaN或inf: {name}，替换为0")
                    g = torch.where(torch.isnan(g) | torch.isinf(g), torch.zeros_like(g), g)
                raw_grads[name] = g.clone()

            # 用 batch_size 作为聚合权重，避免把“单 batch 梯度”当作“全数据梯度”
            n_samples_for_weight = batch_size

        # ===== 原始 FedSGD 模式：遍历本地数据并平均 =====
        else:
            train_loss = 0.0
            train_correct = 0

            # FedSGD：lenet5 和 cifar10cnn_v2 使用 1 epoch，resnet18 使用 3 epochs
            num_local_epochs = 1 if self.model_type != 'resnet18' else 3

            accumulated_grads = {name: torch.zeros_like(param)
                                for name, param in model.named_parameters()}

            for _ in range(num_local_epochs):
                epoch_grad_sums = {name: torch.zeros_like(param)
                                for name, param in model.named_parameters()}
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

                    # 将 batch-mean 梯度按 batch_size 转成 sample-sum，再累加
                    for name, param in model.named_parameters():
                        if param.grad is None:
                            continue
                        g = param.grad.detach()
                        if torch.isnan(g).any() or torch.isinf(g).any():
                            if self.logger:
                                self.logger.warning(f"[Client {self.rank}] 梯度包含NaN或inf: {name}，替换为0")
                            g = torch.where(torch.isnan(g) | torch.isinf(g), torch.zeros_like(g), g)
                        epoch_grad_sums[name] += g * batch_size

                denom = max(epoch_samples, 1)
                for name in epoch_grad_sums:
                    accumulated_grads[name] += epoch_grad_sums[name] / denom

            raw_grads = {name: grad / num_local_epochs for name, grad in accumulated_grads.items()}

            acc_denom = len(self.train_loader.dataset)
            n_samples_for_weight = len(self.train_loader.dataset)

        # ===== 以下保持你原来的“压缩/加密/元数据透传”逻辑 =====

        if enable_compression and enable_compression_all:
            # Step 1: 动态 topk 稀疏化（先压缩选取，再解压回 dense 的稀疏张量）
            sparse_grads = {}
            for name, local_grad in raw_grads.items():
                if normalized_direction is not None and name in normalized_direction:
                    global_grad = normalized_direction[name]
                    values, indices = self.compressor.compress_tensor(
                        local_grad,
                        global_gradient=global_grad
                    )
                    sparse_grad = self.compressor.decompress_tensor(values, indices, local_grad.shape)
                    sparse_grads[name] = sparse_grad
                else:
                    values, indices = self.compressor.compress_tensor(local_grad)
                    sparse_grad = self.compressor.decompress_tensor(values, indices, local_grad.shape)
                    sparse_grads[name] = sparse_grad

            # Step 2: 对稀疏化后的 dense 张量做“加密扰动 + 量化”（按层）
            encrypted_grads = self.compressor.compress_with_encryption(sparse_grads)

            total_params = sum(g.numel() for g in raw_grads.values())
            sparse_params = sum((g != 0).sum().item() for g in sparse_grads.values())
            sparsity_ratio = 1.0 - (sparse_params / total_params) if total_params > 0 else 0.0

            # if self.logger:
            #     self.logger.info(
            #         f"[Rank {self.rank}]  Loss: {train_loss},  Accuracy: {train_correct / acc_denom},  Sparsity: {sparsity_ratio * 100.0}% (已加密压缩)"
            #     )
            print(f"[Rank {self.rank}]  Loss: {train_loss},  Accuracy: {train_correct / acc_denom},  Sparsity: {sparsity_ratio * 100.0}% (已加密压缩)")

            sparse_grads = encrypted_grads
        else:
            sparse_grads = raw_grads
            sparsity_ratio = 0.0
            compression_status = "未压缩未加密"
            print(f"[Rank {self.rank:>2}]  Loss: {train_loss:>8.6f},  Accuracy: {train_correct/acc_denom:>6.4f},  {compression_status}")

        # 本轮是否启用压缩/稀疏这条路径
        round_enabled = bool(enable_compression and enable_compression_all)
        effective_mode = self.compressor.compression_mode if round_enabled else "none"

        # 你这份实现里：只要 round_enabled 就走 encrypt 分支（mask+量化）
        current_encrypted = bool(round_enabled)

        grads = {
            "n_samples": n_samples_for_weight,
            "named_grads": sparse_grads,
            "compression_mode": effective_mode,
            "encrypted": current_encrypted,
            "enable_compression_all": enable_compression_all,
            "enable_compression": enable_compression,
            "fixed_batch": bool(use_fixed_batch),
        }
        return grads



    def __unlearn(self, model, normalized_direction, enable_compression=True):
        """遗忘学习：FedSGD 框架下的客户端级差分遗忘"""


        criterion = nn.CrossEntropyLoss()
        model.train()

        unlearn_loss = 0.0
        unlearn_correct = 0

        num_local_epochs = 1  # 遗忘阶段默认 1 次 pass，避免尺度过大
        print(f"unlearn_mode{self.unlearn_mode}")

        accumulated_grads = {name: torch.zeros_like(param)
                            for name, param in model.named_parameters()}

        for _ in range(num_local_epochs):
            epoch_grad_sums = {name: torch.zeros_like(param)
                            for name, param in model.named_parameters()}
            epoch_samples = 0

            for data, target in self.train_loader:
                data = data.to(self.device)
                target = target.to(self.device)
                batch_size = data.size(0)
                epoch_samples += batch_size

                output = model(data)
                loss = criterion(output, target)
                unlearn_loss += loss.item() * batch_size
                pred = output.argmax(dim=1, keepdim=True)
                unlearn_correct += pred.eq(target.view_as(pred)).sum().item()

                model.zero_grad(set_to_none=True)
                loss.backward()

                for name, param in model.named_parameters():
                    if param.grad is None:
                        continue
                    g = param.grad.detach()
                    if torch.isnan(g).any() or torch.isinf(g).any():
                        if self.logger:
                            self.logger.warning(f'[Client {self.rank}] 遗忘梯度包含NaN或inf: {name}，替换为0')
                        g = torch.where(torch.isnan(g) | torch.isinf(g), torch.zeros_like(g), g)
                    epoch_grad_sums[name] += g * batch_size

            denom = max(epoch_samples, 1)
            for name in epoch_grad_sums:
                # 梯度上升：取负号
                accumulated_grads[name] += -(epoch_grad_sums[name] / denom)

        raw_grads = {name: grad / num_local_epochs for name, grad in accumulated_grads.items()}

        # ===== Step 5: 稀疏化与压缩（协议不变）=====
        if not enable_compression:
            sparse_grads = raw_grads
            sparsity_ratio = 0.0
        else:
            sparse_grads = {}
            for name, local_grad in raw_grads.items():
                if normalized_direction is not None and name in normalized_direction:
                    global_grad = normalized_direction[name]
                    values, indices = self.compressor.compress_tensor(
                        local_grad,
                        global_gradient=global_grad,
                        unlearning_flag=True  # 仅用于数值调节
                    )
                else:
                    values, indices = self.compressor.compress_tensor(local_grad)

                sparse_grad = self.compressor.decompress_tensor(
                    values, indices, local_grad.shape
                )
                sparse_grads[name] = sparse_grad

            encrypted_grads = self.compressor.compress_with_encryption(sparse_grads)

            total_params = sum(g.numel() for g in raw_grads.values())
            sparse_params = sum((g != 0).sum().item() for g in sparse_grads.values())
            sparsity_ratio = 1.0 - sparse_params / total_params

        # print('[Rank {:>2}] Unlearn Loss: {:>4.6f}, Sparsity: {:.2f}%'.format(
        #     self.rank, unlearn_loss / num_batches, sparsity_ratio * 100.0
        # ))
        self.logger.info(f"[Rank {self.rank}] Unlearn Loss: {unlearn_loss }, Sparsity: {sparsity_ratio * 100.0}%")
        grads = {
            'n_samples': len(self.train_loader.dataset),
            'named_grads': encrypted_grads if enable_compression else sparse_grads
        }

        return grads




    def run(self, enable_compression, enable_compression_all, use_fixed_batch=False):
        model,normalized_direction = self.__load_global_model()
        grads = self.__train(model=model,normalized_direction=normalized_direction,enable_compression=enable_compression,enable_compression_all=enable_compression_all, use_fixed_batch=use_fixed_batch)
        torch.save(grads, './cache/grads_{}.pkl'.format(self.rank))

    def unlearn(self,enable_compression=True):
        """执行遗忘学习
        
        使用梯度上升方法进行遗忘学习，并将结果保存
        """
        model, normalized_direction = self.__load_global_model()
        grads = self.__unlearn(model=model, normalized_direction=normalized_direction,enable_compression=enable_compression)
        torch.save(grads, './cache/grads_{}.pkl'.format(self.rank))
