import torch
import numpy as np

from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms


class loader(object):
    def __init__(self, cmd='cifar10', batch_size=64, n_clients=7, alpha=0.5, seed=42, use_augmentation=True):
        """
        Args:
            cmd: 数据集名称 ('cifar10', 'cifar100' 或 'mnist')
            batch_size: 批次大小
            n_clients: 客户端数量
            alpha: 狄利克雷分布参数，控制数据异质性（alpha越小越Non-IID，alpha越大越IID）
            seed: 随机种子，确保可重复性
        """
        self.cmd = cmd
        self.batch_size = batch_size
        self.n_clients = n_clients
        self.alpha = alpha
        self.seed = seed
        self.use_augmentation = use_augmentation
        self.__load_dataset()
        self.__distribute_data_with_dirichlet()

    def __load_dataset(self):
        # mnist
        self.train_mnist = datasets.MNIST('/home/kemove/dataset/mnist',
                                          train=True,
                                          download=True,
                                          transform=transforms.Compose([
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.1307,), (0.3081,))
                                          ]))

        self.test_mnist = datasets.MNIST('/home/kemove/dataset/mnist',
                                         train=False,
                                         download=True,
                                         transform=transforms.Compose([
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.1307,), (0.3081,))
                                         ]))

        # cifar10
        # self.train_cifar10 = datasets.CIFAR10('/home/kemove/dataset/cifar-10-batches-py',
        #                                       train=True,
        #                                       download=True,
        #                                       transform=transforms.Compose([
        #                                           transforms.ToTensor(),
        #                                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        #                                       ]))
        #                                       # CIFAR-10训练数据增强
        # 训练集是否使用随机增强（攻击/可复现实验一般建议关掉）
        if self.use_augmentation:
            self.train_cifar10 = datasets.CIFAR10('/home/kemove/dataset/cifar-10-batches-py',
                                                train=True,
                                                download=True,
                                                transform=transforms.Compose([
                                                    transforms.RandomCrop(32, padding=4),  # 随机裁剪
                                                    transforms.RandomHorizontalFlip(),      # 随机水平翻转
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                                ]))
        else:
            self.train_cifar10 = datasets.CIFAR10('/home/kemove/dataset/cifar-10-batches-py',
                                                train=True,
                                                download=True,
                                                transform=transforms.Compose([
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                                ]))

        self.test_cifar10 = datasets.CIFAR10('/home/kemove/dataset/cifar-10-batches-py',
                                                    train=False,
                                                    download=True,
                                                    transform=transforms.Compose([
                                                        transforms.ToTensor(),
                                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                                    ]))

        # cifar100
        self.train_cifar100 = datasets.CIFAR100('/home/kemove/dataset/cifar-100-python',
                                               train=True,
                                               download=True,
                                               transform=transforms.Compose([
                                                   transforms.ToTensor(),
                                                   transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                               ]))
        self.test_cifar100 = datasets.CIFAR100('/home/kemove/dataset/cifar-100-python',
                                              train=False,
                                              download=True,
                                              transform=transforms.Compose([
                                                  transforms.ToTensor(),
                                                  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                              ]))

        # 选择数据集并确定类别数
        if self.cmd == 'cifar10':
            self.train_dataset = self.train_cifar10
            self.test_dataset = self.test_cifar10
            self.n_classes = 10
        elif self.cmd == 'cifar100':
            self.train_dataset = self.train_cifar100
            self.test_dataset = self.test_cifar100
            self.n_classes = 100
        else:  # mnist
            self.train_dataset = self.train_mnist
            self.test_dataset = self.test_mnist
            self.n_classes = 10

    # def __distribute_data_with_dirichlet(self):
    #     """使用狄利克雷分布分配数据到各个客户端
        
    #     为每个客户端分配训练数据和测试数据，每个客户端的数据分布由狄利克雷分布采样得到。
    #     """
    #     np.random.seed(self.seed)
    #     torch.manual_seed(self.seed)
        
    #     # 使用数据集的类别数（动态获取）
    #     n_classes = self.n_classes
        
    #     # 按类别组织训练数据索引
    #     train_indices_by_class = [[] for _ in range(n_classes)]
    #     for idx, (_, label) in enumerate(self.train_dataset):
    #         train_indices_by_class[label].append(idx)
        
    #     # 按类别组织测试数据索引
    #     test_indices_by_class = [[] for _ in range(n_classes)]
    #     for idx, (_, label) in enumerate(self.test_dataset):
    #         test_indices_by_class[label].append(idx)
        
    #     # 为每个类别生成狄利克雷分布采样
    #     # 对于每个类别，生成一个n_clients维的分布向量
    #     # 该向量表示该类别的数据如何分配到各个客户端
    #     train_client_indices = [[] for _ in range(self.n_clients)]
    #     test_client_indices = [[] for _ in range(self.n_clients)]
        
    #     for class_idx in range(n_classes):
    #         # 使用狄利克雷分布采样，生成该类别的数据分配比例
    #         # alpha参数控制分布的集中程度：alpha越小，分配越不均匀（更Non-IID）
    #         proportions = np.random.dirichlet([self.alpha] * self.n_clients)
            
    #         # 计算该类别的训练数据分配到各客户端的数量
    #         n_train_samples = len(train_indices_by_class[class_idx])
    #         train_counts = (proportions * n_train_samples).astype(int)
    #         # 处理舍入误差，确保所有数据都被分配
    #         train_counts[-1] = n_train_samples - sum(train_counts[:-1])
            
    #         # 计算该类别的测试数据分配到各客户端的数量
    #         n_test_samples = len(test_indices_by_class[class_idx])
    #         test_counts = (proportions * n_test_samples).astype(int)
    #         # 处理舍入误差，确保所有数据都被分配
    #         test_counts[-1] = n_test_samples - sum(test_counts[:-1])
            
    #         # 随机打乱该类别的数据索引
    #         np.random.shuffle(train_indices_by_class[class_idx])
    #         np.random.shuffle(test_indices_by_class[class_idx])
            
    #         # 将数据分配到各个客户端
    #         train_start = 0
    #         test_start = 0
    #         for client_idx in range(self.n_clients):
    #             # 分配训练数据
    #             train_end = train_start + train_counts[client_idx]
    #             train_client_indices[client_idx].extend(
    #                 train_indices_by_class[class_idx][train_start:train_end]
    #             )
    #             train_start = train_end
                
    #             # 分配测试数据
    #             test_end = test_start + test_counts[client_idx]
    #             test_client_indices[client_idx].extend(
    #                 test_indices_by_class[class_idx][test_start:test_end]
    #             )
    #             test_start = test_end
        
    #     # 保存每个客户端的数据索引
    #     self.client_train_indices = train_client_indices
    #     self.client_test_indices = test_client_indices
        
    #     # 打印数据分布统计信息（只显示前10个类别的详细分布，避免输出过长）
    #     print('\n数据分布统计（狄利克雷分布，alpha={}，数据集：{}，类别数：{}）:'.format(
    #         self.alpha, self.cmd, n_classes))
    #     for client_idx in range(self.n_clients):
    #         # 统计每个客户端各类别的数据量
    #         train_class_counts = [0] * n_classes
    #         test_class_counts = [0] * n_classes
            
    #         for idx in train_client_indices[client_idx]:
    #             _, label = self.train_dataset[idx]
    #             train_class_counts[label] += 1
            
    #         for idx in test_client_indices[client_idx]:
    #             _, label = self.test_dataset[idx]
    #             test_class_counts[label] += 1
            
    #         train_total = sum(train_class_counts)
    #         test_total = sum(test_class_counts)
            
    #         # 计算非零类别数（实际拥有的类别数）
    #         train_nonzero_classes = sum(1 for c in train_class_counts if c > 0)
    #         test_nonzero_classes = sum(1 for c in test_class_counts if c > 0)
            
    #         print('  Client {}: 训练数据 {} 条 ({} 个类别), 测试数据 {} 条 ({} 个类别)'.format(
    #             client_idx, train_total, train_nonzero_classes, test_total, test_nonzero_classes))
            
    #         # 对于CIFAR100，只显示前10个类别的分布，避免输出过长
    #         if n_classes > 10:
    #             print('    训练数据类别分布（前10个类别）: {}'.format(train_class_counts[:10]))
    #             print('    测试数据类别分布（前10个类别）: {}'.format(test_class_counts[:10]))
    #         else:
    #             print('    训练数据类别分布: {}'.format(train_class_counts))
    #             print('    测试数据类别分布: {}'.format(test_class_counts))

    def __distribute_data_with_dirichlet(self):
        """使用狄利克雷分布分配数据到各个客户端
        
        为每个客户端分配训练数据和测试数据，每个客户端的数据分布由狄利克雷分布采样得到。
        使用两次狄利克雷采样：一次控制客户端间的类别比例，一次控制每个类别内的数据分布。
        """
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        n_classes = self.n_classes
        
        # 组织数据索引
        train_indices_by_class = [[] for _ in range(n_classes)]
        test_indices_by_class = [[] for _ in range(n_classes)]
        
        for idx, (_, label) in enumerate(self.train_dataset):
            train_indices_by_class[label].append(idx)
        for idx, (_, label) in enumerate(self.test_dataset):
            test_indices_by_class[label].append(idx)
        
        # 打乱每个类别内的数据顺序
        for class_idx in range(n_classes):
            np.random.shuffle(train_indices_by_class[class_idx])
            np.random.shuffle(test_indices_by_class[class_idx])
        
        train_client_indices = [[] for _ in range(self.n_clients)]
        test_client_indices = [[] for _ in range(self.n_clients)]
        
        # 步骤1：为每个客户端生成类别分布（狄利克雷分布）
        # 生成一个 n_clients × n_classes 的矩阵，每行是一个客户端的类别分布
        client_class_dist = np.random.dirichlet(
            [self.alpha] * n_classes,  # 每个类别的浓度参数
            self.n_clients  # 客户端数量
        )
        
        # 归一化，确保每行的和为1
        client_class_dist = client_class_dist / client_class_dist.sum(axis=1, keepdims=True)
        
        # 计算每个客户端应该从每个类别分配的数据量
        for client_idx in range(self.n_clients):
            # 对训练数据
            total_train_needed = int(len(self.train_dataset) / self.n_clients)  # 近似平均分配
            for class_idx in range(n_classes):
                # 该客户端从该类别的分配比例
                proportion = client_class_dist[client_idx, class_idx]
                
                # 计算分配数量
                n_train_to_assign = int(proportion * total_train_needed)
                n_test_to_assign = int(proportion * len(test_indices_by_class[class_idx]))
                
                # 确保不超过可用数据
                n_train_to_assign = min(n_train_to_assign, len(train_indices_by_class[class_idx]))
                n_test_to_assign = min(n_test_to_assign, len(test_indices_by_class[class_idx]))
                
                if n_train_to_assign > 0:
                    # 从该类别的训练数据中取出相应数量的样本
                    assigned_indices = train_indices_by_class[class_idx][:n_train_to_assign]
                    train_client_indices[client_idx].extend(assigned_indices)
                    # 从列表中移除已分配的数据
                    train_indices_by_class[class_idx] = train_indices_by_class[class_idx][n_train_to_assign:]
                
                if n_test_to_assign > 0:
                    # 从该类别的测试数据中取出相应数量的样本
                    assigned_indices = test_indices_by_class[class_idx][:n_test_to_assign]
                    test_client_indices[client_idx].extend(assigned_indices)
                    # 从列表中移除已分配的数据
                    test_indices_by_class[class_idx] = test_indices_by_class[class_idx][n_test_to_assign:]
        
        # 步骤2：处理剩余数据（由于取整可能剩余）
        # 将剩余数据随机分配给客户端
        for class_idx in range(n_classes):
            # 处理剩余训练数据
            while train_indices_by_class[class_idx]:
                for client_idx in range(self.n_clients):
                    if not train_indices_by_class[class_idx]:
                        break
                    train_client_indices[client_idx].append(train_indices_by_class[class_idx].pop())
            
            # 处理剩余测试数据
            while test_indices_by_class[class_idx]:
                for client_idx in range(self.n_clients):
                    if not test_indices_by_class[class_idx]:
                        break
                    test_client_indices[client_idx].append(test_indices_by_class[class_idx].pop())
        
        # 保存结果
        self.client_train_indices = train_client_indices
        self.client_test_indices = test_client_indices
        
        # 统计和打印
        self._print_distribution_stats(train_client_indices, test_client_indices, n_classes)

    def _print_distribution_stats(self, train_indices, test_indices, n_classes):
        """打印数据分布统计"""
        print('\n数据分布统计（狄利克雷分布，alpha={}，数据集：{}，类别数：{}）:'.format(
            self.alpha, self.cmd, n_classes))
        
        for client_idx in range(min(self.n_clients, 10)):  # 只显示前10个客户端
            train_class_counts = np.zeros(n_classes, dtype=int)
            test_class_counts = np.zeros(n_classes, dtype=int)
            
            for idx in train_indices[client_idx]:
                _, label = self.train_dataset[idx]
                train_class_counts[label] += 1
            
            for idx in test_indices[client_idx]:
                _, label = self.test_dataset[idx]
                test_class_counts[label] += 1
            
            train_total = train_class_counts.sum()
            test_total = test_class_counts.sum()
            train_nonzero = (train_class_counts > 0).sum()
            test_nonzero = (test_class_counts > 0).sum()
            
            print(f'  Client {client_idx}: '
                f'训练数据 {train_total} 条 ({train_nonzero} 个类别), '
                f'测试数据 {test_total} 条 ({test_nonzero} 个类别)')
            
            if n_classes > 10:
                print(f'    训练数据类别分布（前10个类别）: {list(train_class_counts[:10])}')
                print(f'    测试数据类别分布（前10个类别）: {list(test_class_counts[:10])}')
            else:
                print(f'    训练数据类别分布: {list(train_class_counts)}')
                print(f'    测试数据类别分布: {list(test_class_counts)}')

    # def get_loader(self, client_rank):
    #     """获取指定客户端的数据加载器
        
    #     Args:
    #         client_rank: 客户端编号（0到n_clients-1）
            
    #     Returns:
    #         tuple: (train_loader, test_loader) 该客户端的训练和测试数据加载器
    #     """
    #     if client_rank < 0 or client_rank >= self.n_clients:
    #         raise ValueError(f"Client rank must be between 0 and {self.n_clients-1}")
        
    #     # 获取该客户端的训练数据索引
    #     train_indices = self.client_train_indices[client_rank]
    #     train_subset = torch.utils.data.Subset(self.train_dataset, train_indices)
    #     train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True)
        
    #     # 获取该客户端的测试数据索引
    #     test_indices = self.client_test_indices[client_rank]
    #     test_subset = torch.utils.data.Subset(self.test_dataset, test_indices)
    #     test_loader = DataLoader(test_subset, batch_size=self.batch_size, shuffle=False)
        
    #     return train_loader, test_loader
    def get_loader(self, client_rank, use_global_test=False):
        """获取指定客户端的数据加载器
        
        Args:
            client_rank: 客户端编号
            use_global_test: 如果True，使用全局测试集而不是客户端自己的测试集
        """
        if client_rank < 0 or client_rank >= self.n_clients:
            raise ValueError(f"Client rank must be between 0 and {self.n_clients-1}")
        
        # 获取该客户端的训练数据索引
        train_indices = self.client_train_indices[client_rank]
        train_subset = torch.utils.data.Subset(self.train_dataset, train_indices)
        train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True)
        
        if use_global_test:
            # 使用全局测试集
            test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
        else:
            # 获取该客户端的测试数据索引
            test_indices = self.client_test_indices[client_rank]
            test_subset = torch.utils.data.Subset(self.test_dataset, test_indices)
            test_loader = DataLoader(test_subset, batch_size=self.batch_size, shuffle=False)
        
        return train_loader, test_loader
    
    def get_loader_legacy(self, rank):
        """保留旧的数据加载方法（向后兼容，如果需要）
        
        Args:
            rank: 要排除的类别列表
            
        Returns:
            tuple: (train_loader, test_loader)
        """
        dataset_indices = []
        difference = list(set(range(10)).difference(set(rank)))
        
        # 按类别组织索引
        indices = [[], [], [], [], [], [], [], [], [], []]
        for index, data in enumerate(self.train_dataset):
            indices[data[1]].append(index)
        
        for i in difference:
            dataset_indices.extend(indices[i])

        dataset = torch.utils.data.Subset(self.train_cifar10, dataset_indices)
        if self.cmd == 'cifar10':
            dataset = torch.utils.data.Subset(self.train_cifar10, dataset_indices)
        elif self.cmd == 'cifar100':
            dataset = torch.utils.data.Subset(self.train_cifar100, dataset_indices)
        else:
            dataset = torch.utils.data.Subset(self.train_mnist, dataset_indices)

        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True)

        return train_loader, test_loader
