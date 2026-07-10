import os
import torch
import torch.nn as nn
import random
from torch.autograd import Variable

from model.lenet import lenet5


class UnlearningModule(object):
    """联邦遗忘学习模块
    
    使用梯度上升方法实现遗忘学习，支持在联邦学习的任意轮次触发。
    遗忘学习采用与正常训练相同的聚合流程：先中间服务器聚合，再中心服务器聚合。
    一旦客户端被选择为遗忘对象，在遗忘后的后续训练中将不再参与训练和聚合。
    """
    
    def __init__(self, n_client, n_server1, client_assignments, logger=None,compression_mode='adaptive'):
        """
        Args:
            n_client: 客户端总数
            n_server1: 中间服务器数量
            client_assignments: 客户端分配列表，每个元素是一个中间服务器负责的客户端rank列表
            logger: 日志记录器
        """
        self.n_client = n_client
        self.n_server1 = n_server1
        self.client_assignments = client_assignments
        self.forgotten_clients = set()  # 已遗忘的客户端集合
        self.logger = logger  # 保存日志器
        os.makedirs('./cache', exist_ok=True)
        self.compression_mode = compression_mode  # 压缩模式
    def select_unlearning_clients(self, unlearning_config, seed=None):
        """根据配置选择要遗忘的客户端
        
        支持两种配置方式：
        1. 指定具体客户端：{server1_rank: [client_ranks]}
        2. 指定每个区域遗忘数量：{server1_rank: num_clients} 或 {server1_rank: {'num': num, 'method': 'random'}}
        
        Args:
            unlearning_config: 遗忘学习配置
                - 方式1（指定具体客户端）: {0: [0, 1], 1: [3, 4]}
                - 方式2（指定数量）: {0: 2, 1: 1} 表示Server1_0选择2个客户端，Server1_1选择1个客户端
                - 方式3（详细配置）: {0: {'num': 2, 'method': 'random'}, 1: {'num': 1, 'method': 'first'}}
                   method可选: 'random'（随机选择）, 'first'（选择前N个）, 'last'（选择后N个）
            seed: 随机种子（用于可重复性）
        
        Returns:
            unlearning_clients: {server1_rank: [client_ranks]} 格式的字典
        """
        unlearning_clients = {}
        
        if seed is not None:
            random.seed(seed)
        
        for server1_rank, config in unlearning_config.items():
            if server1_rank >= len(self.client_assignments):
                warning_msg = f'[Warning] Server1 rank {server1_rank} 不存在，跳过'
                if self.logger:
                    self.logger.warning(warning_msg)
                else:
                    print(warning_msg)
                continue
            
            # 获取该中间服务器负责的客户端列表（排除已遗忘的客户端）
            available_clients = [c for c in self.client_assignments[server1_rank] 
                               if c not in self.forgotten_clients]
            
            if len(available_clients) == 0:
                warning_msg = f'[Warning] Server1 {server1_rank} 没有可用的客户端（可能都已遗忘），跳过'
                if self.logger:
                    self.logger.warning(warning_msg)
                else:
                    print(warning_msg)
                continue
            
            # 解析配置
            if isinstance(config, list):
                # 方式1: 直接指定客户端列表
                selected_clients = [c for c in config if c in available_clients]
                if len(selected_clients) != len(config):
                    warning_msg = f'[Warning] Server1 {server1_rank} 中部分指定的客户端不存在或已遗忘，已过滤'
                    if self.logger:
                        self.logger.warning(warning_msg)
                    else:
                        print(warning_msg)
            elif isinstance(config, int):
                # 方式2: 指定数量，默认随机选择
                num = min(config, len(available_clients))
                selected_clients = random.sample(available_clients, num)
            elif isinstance(config, dict):
                # 方式3: 详细配置
                num = min(config.get('num', 1), len(available_clients))
                method = config.get('method', 'random')
                
                if method == 'random':
                    selected_clients = random.sample(available_clients, num)
                elif method == 'first':
                    selected_clients = available_clients[:num]
                elif method == 'last':
                    selected_clients = available_clients[-num:]
                else:
                    warning_msg = f'[Warning] 未知的选择方法: {method}，使用随机选择'
                    if self.logger:
                        self.logger.warning(warning_msg)
                    else:
                        print(warning_msg)
                    selected_clients = random.sample(available_clients, num)
            else:
                warning_msg = f'[Warning] Server1 {server1_rank} 的配置格式不正确，跳过'
                if self.logger:
                    self.logger.warning(warning_msg)
                else:
                    print(warning_msg)
                continue
            
            if len(selected_clients) > 0:
                unlearning_clients[server1_rank] = selected_clients
                info_msg = f'[Server1 {server1_rank}] 选择遗忘客户端: {selected_clients} (从 {available_clients} 中选择)'
                if self.logger:
                    self.logger.log_unlearning(info_msg)
                else:
                    print(info_msg)
        
        return unlearning_clients
    
    def get_forgotten_clients(self):
        """获取已遗忘的客户端列表
        
        Returns:
            set: 已遗忘客户端的rank集合
        """
        return self.forgotten_clients.copy()
    
    def is_forgotten(self, client_rank):
        """检查客户端是否已被遗忘
        
        Args:
            client_rank: 客户端rank
        
        Returns:
            bool: 如果客户端已被遗忘返回True，否则返回False
        """
        return client_rank in self.forgotten_clients
    
    def perform_unlearning(self, clients, server1s, center_server, unlearning_clients, unlearning_epochs=1):
        """执行遗忘学习以及聚合
        
        Args:
            clients: 客户端列表
            server1s: 中间服务器列表
            center_server: 中心服务器
            unlearning_clients: 需要进行遗忘学习的客户端rank列表，格式为 {server1_rank: [client_ranks]}
                                例如: {0: [0, 1], 1: [3, 4]} 表示server1_0的客户端0和1，server1_1的客户端3和4进行遗忘
            unlearning_epochs: 遗忘学习的轮数（默认1轮）
        
        Returns:
            None
        """
        separator = '='*60
        if self.logger:
            self.logger.log_unlearning(separator)
            self.logger.log_unlearning('开始执行联邦遗忘学习...')
            self.logger.log_unlearning(separator)
        else:
            print('\n' + separator)
            print('开始执行联邦遗忘学习...')
            print(separator)
        
        # 收集所有要遗忘的客户端
        all_unlearning_clients = set()
        for client_ranks in unlearning_clients.values():
            all_unlearning_clients.update(client_ranks)
        
        # 打印遗忘学习配置
        if self.logger:
            self.logger.log_unlearning('\n遗忘学习配置:')
        else:
            print('\n遗忘学习配置:')
        
        total_unlearning_clients = 0
        for server1_rank, client_ranks in unlearning_clients.items():
            info_msg = f'  Server1 {server1_rank}: 客户端 {client_ranks} 进行遗忘学习 (共{len(client_ranks)}个客户端)'
            if self.logger:
                self.logger.log_unlearning(info_msg)
            else:
                print(info_msg)
            total_unlearning_clients += len(client_ranks)
        
        summary_msg1 = f'总遗忘客户端数: {total_unlearning_clients} / {len(clients)}'
        summary_msg2 = f'遗忘学习轮数: {unlearning_epochs}'
        if self.logger:
            self.logger.log_unlearning(summary_msg1)
            self.logger.log_unlearning(summary_msg2)
        else:
            print(summary_msg1)
            print(summary_msg2)
        
        # 执行遗忘学习轮次
        for unlearn_epoch in range(unlearning_epochs):
            epoch_msg = f'\n--- 遗忘学习轮次 {unlearn_epoch + 1}/{unlearning_epochs} ---'
            if self.logger:
                self.logger.log_unlearning(epoch_msg)
            else:
                print(epoch_msg)
            
            # Step 1: 指定客户端进行遗忘学习（梯度上升）
            step1_msg = 'Step 1: 客户端进行遗忘学习（梯度上升）...'
            if self.logger:
                self.logger.log_unlearning(step1_msg)
            else:
                print(step1_msg)
            
            for server1_rank, client_ranks in unlearning_clients.items():
                for client_rank in client_ranks:
                    if 0 <= client_rank < len(clients):
                        clients[client_rank].unlearn(enable_compression=False if self.compression_mode == 'none' else True)
                        client_msg = f'  [Client {client_rank}] 完成遗忘学习'
                        if self.logger:
                            self.logger.log_unlearning(client_msg)
                        else:
                            print(client_msg)
                    else:
                        warning_msg = f'  [Warning] Client {client_rank} 不存在，跳过'
                        if self.logger:
                            self.logger.warning(warning_msg)
                        else:
                            print(warning_msg)
            
            # Step 2: 中间服务器聚合
            step2_msg = 'Step 2: 中间服务器聚合梯度...'
            if self.logger:
                self.logger.log_unlearning(step2_msg)
            else:
                print(step2_msg)
            
            for server1_rank in range(len(server1s)):
                # 检查该中间服务器是否有客户端进行遗忘学习
                if server1_rank in unlearning_clients and len(unlearning_clients[server1_rank]) > 0:
                    # 该中间服务器有客户端进行遗忘学习
                    # 需要混合聚合：遗忘客户端的梯度 + 正常客户端的梯度
                    server1s[server1_rank].aggregate_unlearning_only(
                        unlearning_client_ranks=unlearning_clients[server1_rank],
                        forgotten_clients=self.forgotten_clients
                    )
                else:
                    # 该中间服务器没有客户端进行遗忘学习，只聚合正常梯度
                    skip_msg = f'[Server1 {server1_rank:>2}] 没有遗忘客户端，跳过聚合'
                    if self.logger:
                        self.logger.log_unlearning(skip_msg)
                    else:
                        print(skip_msg)
            
            # Step 3: 中心服务器聚合所有中间服务器的梯度并更新模型
            step3_msg = 'Step 3: 中心服务器聚合并更新模型...'
            if self.logger:
                self.logger.log_unlearning(step3_msg)
            else:
                print(step3_msg)
            center_server.aggregate()
            
            # Step 4: 将更新后的模型下发给所有客户端
            step4_msg = 'Step 4: 下发更新后的模型给所有客户端...'
            if self.logger:
                self.logger.log_unlearning(step4_msg)
            else:
                print(step4_msg)
            self._broadcast_model_to_clients(clients, center_server)
        
        # 标记这些客户端为已遗忘（在遗忘学习完成后）
        self.forgotten_clients.update(all_unlearning_clients)
        
        final_msg1 = f'\n已标记以下客户端为遗忘状态（后续将不再参与训练和聚合）: {sorted(all_unlearning_clients)}'
        final_msg2 = f'当前已遗忘客户端总数: {len(self.forgotten_clients)}'
        final_msg3 = '\n遗忘学习完成！'
        final_msg4 = separator + '\n'
        
        if self.logger:
            self.logger.log_unlearning(final_msg1)
            self.logger.log_unlearning(final_msg2)
            self.logger.log_unlearning(final_msg3)
            self.logger.log_unlearning(final_msg4)
        else:
            print(final_msg1)
            print(final_msg2)
            print(final_msg3)
            print(final_msg4)
    
    def _broadcast_model_to_clients(self, clients, center_server):
        """将中心服务器的模型下发给所有客户端
        
        Args:
            clients: 客户端列表
            center_server: 中心服务器
        """
        # 模型已经保存在 ./cache/global_model_state.pkl
        # 客户端在下次训练时会自动加载最新的模型
        # 这里可以添加显式的模型同步逻辑（如果需要）
        pass
