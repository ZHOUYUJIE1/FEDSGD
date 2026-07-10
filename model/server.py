import os
import torch

from torch import optim
from torch.autograd import Variable

from model.lenet import lenet5, resnet18, cifar10cnn_v2
from topk import TopkCompressor
from compare.cate_compressor import CATECompressor
from model.attacker_modify_fixed import InternalAttacker


class server(object):
    """中心服务器类
    
    作为内部攻击者（诚实且好奇），中心服务器可以对接收到的梯度进行推理攻击：
    1. 对从中间服务器接收的加密梯度进行攻击
    2. 对解密解压缩后的梯度进行攻击
    """
    def __init__(self, size, data_loader, device=None, encryption_key=None, n_class=10, logger=None,
                 enable_attack=False, attack_types=None, save_visualization_epochs=None, save_iteration_steps=None,
                 compression_mode='adaptive', in_dim=3, model_type='lenet5'):
        """
        Args:
            size: 中间服务器数量
            data_loader: 数据加载器
            device: 设备
            encryption_key: 加密密钥
            n_class: 类别数，默认10（CIFAR10/MNIST），CIFAR100需要设置为100
            logger: 日志记录器
            enable_attack: 是否启用内部攻击者功能（默认False）
            attack_types: 攻击类型列表，如 ['gradient_inversion', 'deep_leakage']
            save_visualization_epochs: 要保存可视化结果的轮次列表，None表示所有轮次都保存
            save_iteration_steps: 在优化迭代的哪些步数保存重建结果图片，None表示不保存中间结果
            compression_mode: 与客户端保持一致的压缩模式，用于判断是否需要解密
            in_dim: 输入通道数，默认3（CIFAR），MNIST需要设置为1
            model_type: 模型类型，'lenet5'、'resnet18' 或 'cifar10cnn_v2'，默认 'lenet5'
        """
        # 设备配置：自动检测GPU是否可用
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.size = size
        self.train_loader = data_loader[0]
        self.test_loader = data_loader[1]
        self.path = './cache/global_model_state.pkl'
        self.n_class = n_class
        self.in_dim = in_dim
        self.model_type = model_type
        os.makedirs('./cache', exist_ok=True)
        self.model = self.__init_server()
        # self.optimizer = optim.Adam(self.model.parameters(), lr=1e-2)
        # 根据模型类型使用不同的优化器配置
        if self.model_type == 'resnet18':
            # ResNet18推荐配置
            self.optimizer = optim.SGD(self.model.parameters(), 
                                        lr=0.1,  # 更高的学习率
                                        momentum=0.9,
                                        weight_decay=5e-4)  # 添加权重衰减
            # 添加学习率调度器
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, 
                                                            milestones=[50, 75], 
                                                            gamma=0.1)
            print('ResNet18 优化和学习调度')
        else:
            # LeNet5 和 CIFAR10CNN_V2 配置
            self.optimizer = optim.SGD(self.model.parameters(), 
                                        lr=0.01, 
                                        momentum=0.9)
            self.scheduler = None
            if self.model_type == 'cifar10cnn_v2':
                print('CIFAR10CNN_V2 优化和学习调度')
            else:
                print('LeNet5 优化和学习调度')
        self.accuracy = []
        
        # 初始化压缩器用于解密解压缩（只有中心服务器有密钥）
        self.compressor = TopkCompressor(
            compress_ratio=0.1,  # 默认值，实际不使用
            encryption_key=encryption_key,  # 传入加密密钥
            compression_mode=compression_mode  # 与客户端保持一致，决定是否需要解密
        )
        self.logger = logger  # 保存日志器
        
        # 初始化内部攻击者
        self.enable_attack = enable_attack
        if enable_attack:
            self.attacker = InternalAttacker(
                attacker_type='center_server',
                attacker_id='center_server',
                logger=logger,
                model=self.model,  # 使用中心服务器的模型
                n_class=n_class,
                device=self.device,
                data_loader=data_loader,
                save_visualization_epochs=save_visualization_epochs,
                save_iteration_steps=save_iteration_steps
            )
            self.attack_types = attack_types if attack_types else ['gradient_inversion', 'deep_leakage']
        else:
            self.attacker = None
            self.attack_types = []

    def __init_server(self):
        # 根据模型类型选择模型创建函数
        if self.model_type == 'resnet18':
            model = resnet18(n_class=self.n_class, in_dim=self.in_dim).to(self.device)
        elif self.model_type == 'cifar10cnn_v2':
            model = cifar10cnn_v2(n_class=self.n_class, in_dim=self.in_dim).to(self.device)
        else:
            model = lenet5(n_class=self.n_class, in_dim=self.in_dim).to(self.device)
        torch.save(model.state_dict(), self.path)
        return model

    def __load_grads(self):
        grads_info = []
        for s in range(self.size):
            grads_info.append(torch.load('./cache/grads_agg1_{}.pkl'.format(s), map_location=self.device))
        return grads_info

    @torch.no_grad()
    def recalibrate_bn(self, data_loader, num_batches=20):
        """
        用少量 batch 重新校准 BatchNorm 的 running_mean/var
        """
        was_training = self.model.training
        self.model.train()  # 必须 train 才会更新 BN running stats

        # 只跑 forward，不反传
        for i, (x, _) in enumerate(data_loader):
            if i >= num_batches:
                break
            x = x.to(self.device)
            _ = self.model(x)

        # 还原状态
        self.model.train(was_training)


    @staticmethod
    def __average_grads(grads_info):
        """聚合中间服务器的梯度（求和后除以总样本数）"""
        total_grads = {}
        n_total_samples = 0
        for info in grads_info:
            n_samples = info['n_samples']
            for k, v in info['named_grads'].items():
                if k not in total_grads:
                    # 直接累加梯度（不乘以样本数）
                    total_grads[k] = v.clone()
                else:
                    # 简单求和
                    total_grads[k] += v
            n_total_samples += n_samples
        # 求和后除以中间服务器的总样本数
        gradients = {}
        for k, v in total_grads.items():
            gradients[k] = torch.div(v, n_total_samples)
        return gradients

    @staticmethod
    def __compute_normalized_direction(gradients):
        """计算聚合后梯度的归一化方向（单位向量）
        
        对每个参数分别归一化，而不是用整体范数归一化。
        这样每个参数都是单位向量，便于后续计算余弦相似度。
        
        Args:
            gradients: 字典，包含所有参数的梯度
            
        Returns:
            normalized_direction: 字典，包含归一化后的梯度方向（每个参数都是单位向量）
            grad_norm: 标量，原始梯度的L2范数（所有参数拼接后的整体范数）
        """
        # 计算整体范数（用于记录）
        grad_vector = []
        for k, v in gradients.items():
            grad_vector.append(v.flatten())
        grad_vector = torch.cat(grad_vector)
        grad_norm = torch.norm(grad_vector, p=2)
        
        # 对每个参数分别归一化（使其成为单位向量）
        normalized_direction = {}
        for k, v in gradients.items():
            param_norm = torch.norm(v.flatten(), p=2)
            if param_norm > 1e-10:
                normalized_direction[k] = v / param_norm
            else:
                normalized_direction[k] = torch.zeros_like(v)
        
        return normalized_direction, grad_norm

    def __step(self, gradients,
           clip_norm=100,             # 例如 1.0 / 5.0 / 10.0
           target_grad_norm=None,     # 例如 2.0；不需要就设 None
           eps=1e-12):
        import math
        import torch
        from torch.nn.utils import clip_grad_norm_

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        # --- 1) 计算 param_norm（更新前权重范数） ---
        with torch.no_grad():
            param_sq = 0.0
            for p in self.model.parameters():
                if p is None:
                    continue
                param_sq += float(p.data.norm(2).item() ** 2)
            param_norm = math.sqrt(param_sq)

        # --- 2) 写入 p.grad，并做 NaN/Inf 清零，同时统计 grad_norm ---
        with torch.no_grad():
            grad_sq = 0.0
            nan_inf_elems = 0

            for name, p in self.model.named_parameters():
                if not p.requires_grad:
                    continue

                g = gradients.get(name, None)
                if g is None:
                    # 没有就置零，避免 KeyError
                    p.grad = torch.zeros_like(p.data)
                    continue

                g = g.to(p.data.device)

                bad = torch.isnan(g) | torch.isinf(g)
                if bad.any():
                    nan_inf_elems += int(bad.sum().item())
                    g = g.clone()
                    g[bad] = 0.0

                p.grad = g
                grad_sq += float(g.norm(2).item() ** 2)

            grad_norm = math.sqrt(grad_sq)

        # --- 3) 打印 norm 信息（你要的两个数） ---
        if self.logger:
            self.logger.info(
                f"[center step] param_norm={param_norm:.6f} grad_norm={grad_norm:.6f} nan_inf_elems={nan_inf_elems}"
            )
        else:
            print(f"[center step] param_norm={param_norm:.6f} grad_norm={grad_norm:.6f} nan_inf_elems={nan_inf_elems}")

        # --- 4) 可选：全局缩放（只缩小不放大） ---
        if target_grad_norm is not None:
            with torch.no_grad():
                scale = float(target_grad_norm) / float(grad_norm + eps)
                if scale < 1.0:
                    for p in self.model.parameters():
                        if p.grad is not None:
                            p.grad.mul_(scale)
                    if self.logger:
                        self.logger.warning(f"[center step] global scale grads by {scale:.6f} to target_grad_norm={target_grad_norm}")
                    else:
                        print(f"[center step] global scale grads by {scale:.6f} to target_grad_norm={target_grad_norm}")

        # --- 5) clip（真正对 optimizer 生效） ---
        if clip_norm is not None:
            total_norm_before_clip = clip_grad_norm_(self.model.parameters(), max_norm=float(clip_norm))
            if self.logger:
                self.logger.warning(
                    f"[center step] clip_grad_norm_ max_norm={clip_norm} total_norm_before_clip={float(total_norm_before_clip):.6f}"
                )
            else:
                print(f"[center step] clip_grad_norm_ max_norm={clip_norm} total_norm_before_clip={float(total_norm_before_clip):.6f}")

        # --- 6) step ---
        self.optimizer.step()



    def __test(self):
        test_correct = 0
        self.model.eval()
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = Variable(data).to(self.device), Variable(target).to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1, keepdim=True)
                test_correct += pred.eq(target.view_as(pred)).sum().item()
        return test_correct / len(self.test_loader.dataset)

    def aggregate(self, epoch=None, attack_rounds=None):
        """聚合所有中间服务器的梯度并更新模型
        
        如果启用攻击且当前轮次在攻击轮次列表中，执行推理攻击：
        1. 对从中间服务器接收的加密梯度进行攻击
        2. 对解密解压缩后的梯度进行攻击
        
        Args:
            epoch: 当前训练轮次（用于攻击）
            attack_rounds: 攻击轮次列表，如果为None且enable_attack=True，则在所有轮次攻击
        """
        # Step 1: 加载中间服务器聚合的加密梯度
        grads_info = self.__load_grads()
        
        # 如果启用攻击且当前轮次需要攻击，对加密梯度进行攻击
        if self.enable_attack and self.attacker is not None and epoch is not None:
            should_attack = (attack_rounds is None) or (epoch in attack_rounds)
            if should_attack:
                # 攻击1: 对从中间服务器接收的加密梯度进行攻击
                for s in range(self.size):
                    server1_grads = grads_info[s].get('named_grads', {})
                    attack_results_encrypted = self.attacker.attack_on_gradients(
                        gradients=server1_grads,
                        epoch=epoch,
                        attack_types=self.attack_types
                    )
                    if self.logger:
                        self.logger.log_attack('center_server', 'center_server', epoch, 
                                             attack_results_encrypted, attack_stage='encrypted', server1_rank=s)
        
        # # Step 2: 聚合加密梯度（仍然是加密状态）
        # encrypted_gradients = self.__average_grads(grads_info)
        
        # # Step 3: 获取模型参数形状（用于解压缩）
        # original_shapes = {k: v.shape for k, v in self.model.named_parameters()}
        
        # # Step 4: 解密解压缩（只有中心服务器有密钥）
        # gradients = self.compressor.decompress_with_decryption(
        #     encrypted_gradients,
        #     original_shapes,
        #     encrypted=(self.compressor.compression_mode != 'none' and self.compressor.encryption_key is not None)
        # )
                # ===== 新增：按“本轮 payload”决定是否解密，而不是按全局配置猜 =====
        if len(grads_info) == 0:
            raise ValueError("[center_server] grads_info is empty")

        round_mode = grads_info[0].get("compression_mode", self.compressor.compression_mode)
        round_encrypted = bool(grads_info[0].get("encrypted", False))

        # --- server.py: aggregate() 里，在 round_mode/round_encrypted 解析后加入 ---

        # 1) 元数据必须存在（避免默认值吞错）
        for i, info in enumerate(grads_info):
            if "compression_mode" not in info or "encrypted" not in info:
                raise AssertionError(
                    f"[center_server] server1 payload missing fields at idx={i}. "
                    f"Keys={list(info.keys())}. Must include compression_mode/encrypted."
                )

        # 2) 强规则：mode != 'none' 必须 encrypted=True
        if round_mode != "none" and not round_encrypted:
            raise AssertionError(
                f"[center_server] round_mode={round_mode} requires encrypted=True, but got encrypted=False. "
                f"This would skip decryption and corrupt training."
            )

        # 3) encrypted=True 必须有 key
        if round_encrypted and self.compressor.encryption_key is None:
            raise AssertionError("[center_server] encrypted=True but encryption_key is None (cannot decrypt).")


        for info in grads_info[1:]:
            m = info.get("compression_mode", self.compressor.compression_mode)
            e = bool(info.get("encrypted", False))
            if m != round_mode or e != round_encrypted:
                raise ValueError(
                    f"[center_server] Mixed compression/encryption from server1 in the same round: "
                    f"expected (mode={round_mode}, encrypted={round_encrypted}) "
                    f"but got (mode={m}, encrypted={e}). "
                    f"Fix server1/client so the round is consistent."
                )

        # Step 2: 聚合梯度（此时仍可能是“加密态”，但已经是dense张量了）
        encrypted_gradients = self.__average_grads(grads_info)

        # Step 3: 决定是否解密（严格按 round_encrypted）
        need_decrypt = round_encrypted

        if need_decrypt and self.compressor.encryption_key is None:
            raise ValueError("[center_server] round_encrypted=True but encryption_key is None (cannot decrypt).")

        if self.logger:
            self.logger.info(
                f"Round meta from payload: compression_mode={round_mode}, encrypted={need_decrypt}"
            )

        if need_decrypt:
            if self.logger:
                self.logger.info("已解密")

            original_shapes = {k: v.shape for k, v in self.model.named_parameters()}
            gradients = self.compressor.decompress_with_decryption(
                encrypted_gradients,
                original_shapes,
                encrypted=True
            )
        else:
            gradients = encrypted_gradients
            if self.logger:
                self.logger.info("无需解密或未解密")
        
        # 如果启用攻击且当前轮次需要攻击，对解密后的梯度进行攻击
        if self.enable_attack and self.attacker is not None and epoch is not None:
            should_attack = (attack_rounds is None) or (epoch in attack_rounds)
            if should_attack:
                # 攻击2: 对解密解压缩后的梯度进行攻击（可以完全重建）
                # 注意：这里可以获取原始数据用于对比（如果可能）
                attack_results_decrypted = self.attacker.attack_on_gradients(
                    gradients=gradients,
                    epoch=epoch,
                    attack_types=self.attack_types
                )
                if self.logger:
                    self.logger.log_attack('center_server', 'center_server', epoch, 
                                         attack_results_decrypted, attack_stage='decrypted')
        
        # 检查梯度是否包含NaN或inf
        has_nan = False
        has_inf = False
        for k, v in gradients.items():
            if torch.isnan(v).any():
                has_nan = True
                if self.logger:
                    self.logger.warning(f'梯度包含NaN: {k}')
            if torch.isinf(v).any():
                has_inf = True
                if self.logger:
                    self.logger.warning(f'梯度包含inf: {k}')
        
        # 如果梯度异常，进行梯度裁剪
        if has_nan or has_inf:
            if self.logger:
                self.logger.warning('检测到梯度异常（NaN或inf），进行梯度裁剪...')
            # 将NaN和inf替换为0
            for k, v in gradients.items():
                gradients[k] = torch.where(torch.isnan(v) | torch.isinf(v), 
                                          torch.zeros_like(v), v)
        
        # 梯度裁剪：防止梯度爆炸
        # max_grad_norm = 10.0  # 最大梯度范数
        # grad_norm = 0.0
        # for k, v in gradients.items():
        #     param_norm = torch.norm(v.flatten(), p=2)
        #     grad_norm += param_norm.item() ** 2
        # grad_norm = grad_norm ** 0.5
        
        # if grad_norm > max_grad_norm:
        #     if self.logger:
        #         self.logger.warning(f'梯度范数过大 ({grad_norm:.2f})，进行裁剪到 {max_grad_norm}')
        #     clip_coef = max_grad_norm / (grad_norm + 1e-6)
        #     for k, v in gradients.items():
        #         gradients[k] = v * clip_coef
        
        # 计算聚合后梯度的归一化方向
        normalized_direction, grad_norm = self.__compute_normalized_direction(gradients)
        
        if self.logger:
            self.logger.log_gradient_info(grad_norm.item())
        else:
            print('[Gradient Info]  Gradient L2 Norm: {:.6f}'.format(grad_norm.item()))
        
        # 保存归一化方向时，确保所有张量都在CPU上（便于跨设备加载）
        normalized_direction_cpu = {k: v.cpu() if isinstance(v, torch.Tensor) else v 
                                    for k, v in normalized_direction.items()}
        torch.save(normalized_direction_cpu, './cache/normalized_direction.pkl')

        self.__step(gradients)
        # 在self.__step(gradients)之后执行学习率调度器
        if self.scheduler is not None:
            self.scheduler.step()
        # 只在ResNet18时重新校准BatchNorm（CIFAR10CNN_V2和LeNet5没有BatchNorm层）
        if self.model_type == 'resnet18' and self.train_loader is not None:
            self.recalibrate_bn(self.train_loader, num_batches=50)

        torch.save(self.model.state_dict(), './cache/global_model_state.pkl')

        test_accuracy = self.__test()
        self.accuracy.append(test_accuracy)
        torch.save(self.accuracy, './cache/accuracy.pkl')
        
        # 实时保存准确率到txt文件
        accuracy_file = './cache/center_server_accuracy.txt'
        # 如果是第一次，写入表头
        if len(self.accuracy) == 1:
            with open(accuracy_file, 'w', encoding='utf-8') as f:
                f.write("Iteration\tAccuracy\n")
        
        # 追加当前轮次的准确率
        with open(accuracy_file, 'a', encoding='utf-8') as f:
            f.write(f"{len(self.accuracy)}\t{test_accuracy:.6f}\n")
        
        if self.logger:
            self.logger.info(f'\n[Global Model]  Test Accuracy: {test_accuracy * 100.:.2f}%\n')
        else:
            print('\n[Global Model]  Test Accuracy: {:.2f}%\n'.format(test_accuracy * 100.))
