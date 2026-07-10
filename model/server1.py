import torch
import os

from model.attacker_modify_fixed import InternalAttacker


class server1(object):
    """中间服务器类，负责聚合其区域内客户端的梯度
    
    每个中间服务器负责一个区域，该区域包含多个客户端。
    中间服务器聚合区域内所有客户端的梯度，并将结果保存供中心服务器使用。
    已遗忘的客户端将不再参与聚合。
    
    作为内部攻击者（诚实且好奇），中间服务器可以对接收到的梯度进行推理攻击。
    """
    def __init__(self, rank, client_ranks, logger=None, enable_attack=False, attack_types=None,
                 model=None, n_class=10, device=None, data_loader=None, save_visualization_epochs=None,
                 save_iteration_steps=None):
        """
        Args:
            rank: 中间服务器的标识符（rank）
            client_ranks: 该中间服务器负责的客户端rank列表
            logger: 日志记录器
            enable_attack: 是否启用内部攻击者功能（默认False）
            attack_types: 攻击类型列表，如 ['gradient_inversion', 'deep_leakage']
            model: 模型实例（用于梯度反转攻击），应与全局模型类型一致（lenet5、resnet18 或 cifar10cnn_v2）
            n_class: 类别数
            device: 设备
            data_loader: 数据加载器
            save_visualization_epochs: 要保存可视化结果的轮次列表，None表示所有轮次都保存
            save_iteration_steps: 在优化迭代的哪些步数保存重建结果图片，None表示不保存中间结果
        """
        self.rank = rank
        self.client_ranks = client_ranks
        self.output_path = './cache/grads_agg1_{}.pkl'.format(self.rank)
        self.logger = logger  # 保存日志器
        os.makedirs('./cache', exist_ok=True)
        
        # 初始化内部攻击者
        self.enable_attack = enable_attack
        if enable_attack:
            self.attacker = InternalAttacker(
                attacker_type='intermediate_server',
                attacker_id=f'server1_{rank}',
                logger=logger,
                model=model,
                n_class=n_class,
                device=device,
                data_loader=data_loader,
                save_visualization_epochs=save_visualization_epochs,
                save_iteration_steps=save_iteration_steps
            )
            self.attack_types = attack_types if attack_types else ['gradient_inversion', 'deep_leakage']
        else:
            self.attacker = None
            self.attack_types = []

    # def __load_client_grads(self, forgotten_clients=None):
    #     """加载该中间服务器负责的所有客户端的梯度（排除已遗忘的客户端）
        
    #     Args:
    #         forgotten_clients: 已遗忘的客户端集合，如果为None则加载所有客户端
        
    #     Returns:
    #         grads_info: 包含所有客户端梯度信息的列表
    #     """
    #     if forgotten_clients is None:
    #         forgotten_clients = set()
        
    #     grads_info = []
    #     active_clients = []
    #     for client_rank in self.client_ranks:
    #         # 跳过已遗忘的客户端
    #         if client_rank in forgotten_clients:
    #             continue
            
    #         try:
    #             grad_info = torch.load('./cache/grads_{}.pkl'.format(client_rank))
    #             grads_info.append(grad_info)
    #             active_clients.append(client_rank)
    #         except FileNotFoundError:
    #             print('[Server1 {:>2}] Warning: Gradients from client {} not found, skipping.'.format(
    #                 self.rank, client_rank))
        
    #     return grads_info, active_clients
    def __load_client_grads(self, forgotten_clients=None, only_unlearning_clients=None):
        """加载该中间服务器负责的所有客户端的压缩梯度
        
        Args:
            forgotten_clients: 已遗忘的客户端集合，如果为None则加载所有客户端
            only_unlearning_clients: 如果指定，只加载这些客户端的梯度（用于遗忘学习）
        
        Returns:
            grads_info: 包含所有客户端压缩梯度信息的列表
        """
        if forgotten_clients is None:
            forgotten_clients = set()
        
        grads_info = []
        active_clients = []
        
        # 确定要加载的客户端列表
        if only_unlearning_clients is not None:
            # 只加载指定的遗忘客户端
            target_clients = [c for c in only_unlearning_clients if c in self.client_ranks]
        else:
            # 加载所有客户端（排除已遗忘的）
            target_clients = [c for c in self.client_ranks if c not in forgotten_clients]
        
        for client_rank in target_clients:
            try:
                # 加载梯度时，使用 map_location='cpu' 确保兼容性
                # 梯度聚合可以在CPU上进行，后续由server加载到正确设备
                grad_info = torch.load('./cache/grads_{}.pkl'.format(client_rank), map_location='cpu')
                grads_info.append(grad_info)
                active_clients.append(client_rank)
            except FileNotFoundError:
                warning_msg = f'[Server1 {self.rank:>2}] Warning: Gradients from client {client_rank} not found, skipping.'
                if self.logger:
                    self.logger.warning(warning_msg)
                else:
                    print(warning_msg)
        
        return grads_info, active_clients

    @staticmethod
    def __average_grads(grads_info):
        """聚合多个客户端的梯度（简单求和，不平均）
        
        注意：这里聚合的是加密压缩后的梯度，中间服务器无法解密
        
        Args:
            grads_info: 包含客户端梯度信息的列表，每个元素包含 'n_samples' 和 'named_grads'
                       其中 'named_grads' 是加密压缩后的梯度
            
        Returns:
            aggregated_grads: 聚合后的梯度字典，包含 'n_samples' 和 'named_grads'
                             仍然是加密压缩状态
        """
        if len(grads_info) == 0:
            raise ValueError("No gradient information available for aggregation")
        
        total_grads = {}
        n_total_samples = 0
        
        for info in grads_info:
            n_samples = info['n_samples']
            for k, v in info['named_grads'].items():
                if k not in total_grads:
                    # 初始化时乘以样本数
                    total_grads[k] = v * n_samples
                else:
                    # 累加加密压缩后的梯度（按样本数加权）
                    total_grads[k] += v * n_samples
            n_total_samples += n_samples
        
        # 不进行平均，直接返回求和结果（仍然是加密压缩状态）
        aggregated_grads = total_grads
        
                # ===== 新增：透传并校验本轮 metadata（encrypted / compression_mode） =====
        if len(grads_info) == 0:
            raise ValueError("[server1] grads_info is empty")

        # --- server1.py: __average_grads(grads_info) 里，在取 round_mode/round_encrypted 前加 ---

        # 1) 硬要求字段必须存在
        for i, info in enumerate(grads_info):
            if "compression_mode" not in info or "encrypted" not in info:
                raise AssertionError(
                    f"[server1] Client payload missing required fields at idx={i}. "
                    f"Keys={list(info.keys())}. "
                    f"Client MUST send 'compression_mode' and 'encrypted'."
                )

        # 2) 严格读本轮元数据
        round_mode = info0_mode = grads_info[0]["compression_mode"]
        round_encrypted = info0_enc = bool(grads_info[0]["encrypted"])

        # 3) 本轮一致性硬断言（你原来就有，但这里不再允许缺省）
        for j, info in enumerate(grads_info[1:], start=1):
            m = info["compression_mode"]
            e = bool(info["encrypted"])
            if m != round_mode or e != round_encrypted:
                raise AssertionError(
                    f"[server1] Mixed compression/encryption in same round. "
                    f"expected (mode={round_mode}, encrypted={round_encrypted}) "
                    f"but idx={j} got (mode={m}, encrypted={e})."
                )

        # 4) 强规则：只要 mode != 'none'，encrypted 必须为 True
        if round_mode != "none" and not round_encrypted:
            raise AssertionError(
                f"[server1] round_mode={round_mode} requires encrypted=True, but got encrypted=False. "
                f"This would cause center_server to skip decryption and train on masked gradients."
            )


        for info in grads_info[1:]:
            m = info.get("compression_mode", "none")
            e = bool(info.get("encrypted", False))
            if m != round_mode or e != round_encrypted:
                raise ValueError(
                    f"[server1] Mixed compression/encryption in the same round: "
                    f"expected (mode={round_mode}, encrypted={round_encrypted}) "
                    f"but got (mode={m}, encrypted={e}). "
                    f"Fix client-side logic so all active clients in a round match."
                )

        return {
            "n_samples": n_total_samples,
            "named_grads": aggregated_grads,
            "compression_mode": round_mode,
            "encrypted": round_encrypted,
        }


    def aggregate(self, forgotten_clients=None, epoch=None, attack_rounds=None):
        """聚合该中间服务器负责的所有客户端的梯度，并保存结果
        
        该方法会：
        1. 加载区域内所有客户端的梯度（排除已遗忘的客户端）
        2. 按样本数加权平均聚合梯度
        3. 将聚合结果保存到 './cache/grads_agg1_{rank}.pkl'
        4. 如果启用攻击且当前轮次在攻击轮次列表中，执行推理攻击
        
        Args:
            forgotten_clients: 已遗忘的客户端集合，如果为None则聚合所有客户端
            epoch: 当前训练轮次（用于攻击）
            attack_rounds: 攻击轮次列表，如果为None且enable_attack=True，则在所有轮次攻击
        """
        if forgotten_clients is None:
            forgotten_clients = set()
        
        # 加载客户端梯度（排除已遗忘的客户端）
        grads_info, active_clients = self.__load_client_grads(forgotten_clients)
        
        if len(grads_info) == 0:
            warning_msg = f'[Server1 {self.rank:>2}] No gradients available (all clients forgotten or no gradients), skipping aggregation.'
            if self.logger:
                self.logger.warning(warning_msg)
            else:
                print(warning_msg)
            return
        
        # 如果启用攻击且当前轮次需要攻击，对每个客户端的梯度进行攻击
        if self.enable_attack and self.attacker is not None and epoch is not None:
            should_attack = (attack_rounds is None) or (epoch in attack_rounds)
            if should_attack:
                for idx, grad_info in enumerate(grads_info):
                    client_grads = grad_info.get('named_grads', {})
                    client_rank = active_clients[idx] if idx < len(active_clients) else None
                    # 注意：中间服务器看到的是加密压缩的梯度，可能无法完全重建
                    # 但可以尝试攻击
                    attack_results = self.attacker.attack_on_gradients(
                        gradients=client_grads,
                        epoch=epoch,
                        client_rank=client_rank,
                        attack_types=self.attack_types
                    )
                    # 记录攻击结果
                    if self.logger:
                        self.logger.log_attack('intermediate_server', self.rank, epoch, attack_results)
        
        # 聚合梯度
        aggregated_grads = self.__average_grads(grads_info)
        
        # 保存聚合后的梯度
        torch.save(aggregated_grads, self.output_path)
        
        forgotten_count = len([c for c in self.client_ranks if c in forgotten_clients])
        
        # 使用日志记录聚合信息
        if self.logger:
            self.logger.log_server_aggregate('intermediate', {
                'rank': self.rank,
                'active_clients': len(active_clients),
                'forgotten_clients': forgotten_count,
                'total_samples': aggregated_grads['n_samples']
            })
        else:
            print('[Server1 {:>2}] Aggregated gradients from {} active clients ({} forgotten, total samples: {})'.format(
                self.rank,
                len(active_clients),
                forgotten_count,
                aggregated_grads['n_samples']
            ))
    
    def aggregate_unlearning_only(self, unlearning_client_ranks, forgotten_clients=None):
        """只聚合遗忘客户端的梯度
        
        Args:
            unlearning_client_ranks: 进行遗忘学习的客户端rank列表
            forgotten_clients: 已遗忘的客户端集合
        """
        if forgotten_clients is None:
            forgotten_clients = set()
        
        # 只加载遗忘客户端的压缩梯度
        grads_info, active_clients = self.__load_client_grads(
            forgotten_clients=forgotten_clients,
            only_unlearning_clients=unlearning_client_ranks
        )
        
        if len(grads_info) == 0:
            warning_msg = f'[Server1 {self.rank:>2}] No gradients available (all clients forgotten or no gradients), skipping aggregation.'
            if self.logger:
                self.logger.warning(warning_msg)
            else:
                print(warning_msg)
            return
        
        # 聚合梯度
        aggregated_grads = self.__average_grads(grads_info)
        
        # 保存聚合后的梯度
        torch.save(aggregated_grads, self.output_path)
        
        forgotten_count = len([c for c in self.client_ranks if c in forgotten_clients])
        
        if self.logger:
            self.logger.info(
                f'[Server1 {self.rank:>2}] Aggregated unlearning gradients from {len(active_clients)} clients '
                f'({forgotten_count} forgotten, total samples: {aggregated_grads["n_samples"]})'
            )
        else:
            print('[Server1 {:>2}] Aggregated gradients from {} active clients ({} forgotten, total samples: {})'.format(
                self.rank,
                len(active_clients),
                forgotten_count,
                aggregated_grads['n_samples']
            ))

    def aggregate_unlearning(self):
        """聚合该中间服务器负责的客户端的遗忘梯度
        
        与正常聚合方法相同，但用于遗忘学习场景
        """
        # 复用正常的聚合逻辑（遗忘梯度也是梯度，聚合方式相同）
        self.aggregate()
        info_msg = f'[Server1 {self.rank:>2}] Aggregated unlearning gradients'
        if self.logger:
            self.logger.info(info_msg)
        else:
            print(info_msg)

    def aggregate_mixed(self, unlearning_client_ranks, forgotten_clients=None):
        """混合聚合：同时处理遗忘客户端和正常客户端的梯度
        
        Args:
            unlearning_client_ranks: 进行遗忘学习的客户端rank列表
            forgotten_clients: 已遗忘的客户端集合（排除这些客户端）
        """
        if forgotten_clients is None:
            forgotten_clients = set()
        
        # 加载所有客户端的梯度（包括遗忘和正常，但排除已遗忘的客户端）
        grads_info, active_clients = self.__load_client_grads(forgotten_clients)
        
        if len(grads_info) == 0:
            warning_msg = f'[Server1 {self.rank:>2}] No gradients available, skipping aggregation.'
            if self.logger:
                self.logger.warning(warning_msg)
            else:
                print(warning_msg)
            return
        
        # 聚合所有梯度（遗忘梯度和正常梯度混合聚合）
        aggregated_grads = self.__average_grads(grads_info)
        
        # 保存聚合后的梯度
        torch.save(aggregated_grads, self.output_path)
        
        # 统计信息
        unlearning_count = len([c for c in active_clients if c in unlearning_client_ranks])
        normal_count = len(active_clients) - unlearning_count
        forgotten_count = len([c for c in self.client_ranks if c in forgotten_clients])
        
        info_msg = (
            f'[Server1 {self.rank:>2}] Mixed aggregation: {unlearning_count} unlearning + '
            f'{normal_count} normal clients ({forgotten_count} forgotten, total samples: {aggregated_grads["n_samples"]})'
        )
        if self.logger:
            self.logger.info(info_msg)
        else:
            print(info_msg)

