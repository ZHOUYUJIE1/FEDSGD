import random
import torch
from torch.utils.data import DataLoader
import logging
import os

from model.data import loader
from model.server import server
from model.server1 import server1
from model.client import client
from model.plot import plot
from model.Unlearning import UnlearningModule
from model.logger import get_logger
from model.attacker_modify_fixed import ExternalAttacker


def federated_learning():

    import torch
    print("cuda available:", torch.cuda.is_available())
    print("device count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("current device:", torch.cuda.current_device())
        print("device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
        print("cudnn enabled:", torch.backends.cudnn.enabled)
        print("cudnn version:", torch.backends.cudnn.version())

    # 设备配置：自动检测GPU是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 初始化日志系统
    logger = get_logger(log_dir='./log', log_name='fedsgd', level=logging.INFO)
    
    logger.info(f'使用设备: {device}')
    if torch.cuda.is_available():
        logger.info(f'GPU设备名称: {torch.cuda.get_device_name(0)}')
        logger.info(f'GPU数量: {torch.cuda.device_count()}')
    logger.info('-' * 60)
    
    # 压缩密钥配置（只有客户端和中心服务器知道）
    ENCRYPTION_KEY = "fed_sgd_secret_key_2024"  # 可以改为更安全的密钥
    
    # hyper parameter
    n_client = 50
    n_server1 = 5  # 中间服务器数量
    n_epoch = 200
    batch_size = 64
    use_fixed_batch=False #设置为True为测攻击防御的一个batch的效果，false为测全局收敛性效果
    # 数据分布配置
    dirichlet_alpha = 0.3  # 狄利克雷分布参数，控制数据异质性
                          # alpha越小，数据分布越不均匀（更Non-IID）
                          # alpha越大，数据分布越均匀（更接近IID）
                          # 推荐值：0.1-1.0，0.5是一个中等Non-IID程度
    

    # dataset - 使用狄利克雷分布分配数据
    # 数据集选择配置：'mnist', 'cifar10', 'cifar100'
    logger.info('Initialize Dataset with Dirichlet Distribution...')
    dataset_name = 'mnist'  # 可选：'mnist', 'cifar10', 'cifar100'
    # data_loader = loader(dataset_name, batch_size=batch_size, n_clients=n_client, alpha=dirichlet_alpha, seed=42)
    data_loader = loader(dataset_name, batch_size=batch_size, n_clients=n_client, alpha=dirichlet_alpha, seed=42, use_augmentation=False)

    # # 模型选择配置：'lenet5'、'resnet18' 或 'cifar10cnn_v2'
    if dataset_name == 'cifar100':
        model_type = 'resnet18'
    elif dataset_name == 'cifar10':
        model_type = 'cifar10cnn_v2'
    else:
        model_type = 'lenet5'
    
    # model_type = 'resnet18'  # 可选：'lenet5' 或 'resnet18'
    
    
    
    # 压缩配置
    compress_first_round = False  # 第一轮是否进行压缩（False=不压缩，True=使用固定压缩率）
    compression_mode = 'adaptive'  # 压缩模式：'adaptive'（动态稀疏化）、'fixed'（固定稀疏度）、'none'（不压缩）

    # 攻击者配置
    enable_external_attacker = False  # 是否启用外部攻击者
    enable_internal_attacker_server1 = False  # 是否启用中间服务器内部攻击者
    enable_internal_attacker_center = False  # 是否启用中心服务器内部攻击者
    attack_rounds = [2,50]  # 攻击轮次列表，None表示所有轮次都攻击，例如 [1, 2, 3] 表示只在第1,2,3轮攻击
    attack_types = ['gradient_inversion', 'deep_leakage']  # 攻击类型列表：梯度反转和梯度深度泄露
    save_visualization_epochs = attack_rounds # 保存重建图像对比图的轮次列表，None表示所有轮次都保存，例如 [1, 4] 表示只在第1,4轮保存对比图
    save_iteration_steps = [1, 50, 200,2000,3000]  # 在优化迭代的哪些步数保存重建结果图片，None表示不保存中间结果，例如 [100, 500, 1000] 表示在第100,500,1000次迭代时保存

    # 遗忘学习配置
    unlearning_round =40   # 在第几轮进行遗忘学习（从1开始计数，0表示不进行遗忘学习）
    unlearning_epochs = 1  # 遗忘学习的轮数
    unlearn_mode='unlearn' #遗忘的逻辑，可以为'unlearn'或'invert'
    
    # 方式1: 直接指定要遗忘的客户端（精确控制）
    unlearning_config = {
        # 0: [0, 1],  # Server1_0的客户端0和1进行遗忘学习
        1: [2],  # Server1_1的客户端3和4进行遗忘学习
        # 2: [7, 8],  # Server1_2的客户端7和8进行遗忘学习（可选）
    }
    
    # 方式2: 指定每个区域要遗忘的客户端数量（自动随机选择）
    # unlearning_config = {
    #     # 0: 1,  # Server1_0随机选择2个客户端进行遗忘
    #     1: 1,  # Server1_1随机选择1个客户端进行遗忘
    #     # 2: 0,  # Server1_2不进行遗忘（可以省略）
    # }
    
    # 方式3: 详细配置（指定数量和选择方法）
    # unlearning_config = {
    #     0: {'num': 2, 'method': 'random'},  # Server1_0随机选择2个
    #     1: {'num': 1, 'method': 'first'},   # Server1_1选择前1个
    #     2: {'num': 1, 'method': 'last'},    # Server1_2选择后1个
    # }

    


    # 记录实验配置
    config = {
        'n_client': n_client,
        'n_server1': n_server1,
        'n_epoch': n_epoch,
        'batch_size': batch_size,
        '模型类型': model_type,
        '数据集': dataset_name,
        '迪利克雷值': dirichlet_alpha,
        'compress_first_round': compress_first_round,
        'compression_mode': compression_mode,
        'enable_external_attacker': enable_external_attacker,
        'enable_internal_attacker_server1': enable_internal_attacker_server1,
        'enable_internal_attacker_center': enable_internal_attacker_center,
        'attack_rounds': attack_rounds,
        'attack_types': attack_types,
        'save_visualization_epochs': save_visualization_epochs,
        'save_iteration_steps': save_iteration_steps,
        'unlearning_round': unlearning_round,
        'unlearning_epochs': unlearning_epochs,
        'unlearning_config': unlearning_config
        
    }
    logger.log_config(config)
    # 获取数据集类别数和输入通道数
    n_class = data_loader.n_classes
    # 根据数据集类型确定输入通道数：MNIST=1, CIFAR=3
    if dataset_name == 'mnist':
        in_dim = 1
    else:  # cifar10 or cifar100
        in_dim = 3
    logger.info(f'数据集类别数: {n_class}, 输入通道数: {in_dim}')
    logger.info(f'使用模型: {model_type}')

    # 将客户端分配给中间服务器
    # 每个中间服务器负责一部分客户端
    clients_per_server1 = n_client // n_server1
    client_assignments = []
    for i in range(n_server1):
        start_idx = i * clients_per_server1
        if i == n_server1 - 1:
            # 最后一个中间服务器负责剩余的客户端
            end_idx = n_client
        else:
            end_idx = (i + 1) * clients_per_server1
        client_assignments.append(list(range(start_idx, end_idx)))
    
    logger.info('\nClient assignments to Server1:')
    for i, assignment in enumerate(client_assignments):
        logger.info(f'  Server1 {i}: clients {assignment}')

    # 准备中心服务器测试集：使用全体测试数据
    full_train_loader = DataLoader(data_loader.train_dataset, batch_size=batch_size, shuffle=True)
    full_test_loader = DataLoader(data_loader.test_dataset, batch_size=batch_size, shuffle=False)
    
    # initialize center server (中心服务器) - 传入密钥和类别数
    logger.info('\nInitialize Center Server...')
    center_server = server(
        size=n_server1, 
        data_loader=(full_train_loader, full_test_loader),  # 使用全体测试集进行评估
        device=device,
        encryption_key=ENCRYPTION_KEY,  # 传入密钥
        n_class=n_class,  # 传入类别数
        logger=logger,  # 传入日志器
        enable_attack=enable_internal_attacker_center,  # 启用内部攻击者
        attack_types=attack_types,
        save_visualization_epochs=save_visualization_epochs,
        save_iteration_steps=save_iteration_steps,
        compression_mode=compression_mode,  # 与客户端保持一致，用于判断是否需要解密
        in_dim=in_dim,  # 传入输入通道数
        model_type=model_type  # 传入模型类型
    )

    # initialize intermediate servers (中间服务器)
    logger.info('Initialize Intermediate Servers (Server1)...')
    # 为中间服务器创建模型（用于攻击）
    from model.lenet import lenet5, resnet18, cifar10cnn_v2
    if model_type == 'resnet18':
        server1_model = resnet18(n_class=n_class, in_dim=in_dim).to(device)
    elif model_type == 'cifar10cnn_v2':
        server1_model = cifar10cnn_v2(n_class=n_class, in_dim=in_dim).to(device)
    else:
        server1_model = lenet5(n_class=n_class, in_dim=in_dim).to(device)
    if os.path.exists('./cache/global_model_state.pkl'):
        try:
            server1_model.load_state_dict(torch.load('./cache/global_model_state.pkl', map_location=device))
        except:
            pass
    
    server1s = []
    for i in range(n_server1):
        server1s.append(server1(
            rank=i, 
            client_ranks=client_assignments[i], 
            logger=logger,
            enable_attack=enable_internal_attacker_server1,  # 启用内部攻击者
            attack_types=attack_types,
            model=server1_model,  # 传递模型
            n_class=n_class,
            device=device,
            data_loader=data_loader,
            save_visualization_epochs=save_visualization_epochs,
            save_iteration_steps=save_iteration_steps
        ))

    # initialize clients - 传入密钥和类别数，每个客户端使用自己的数据
    logger.info('Initialize Clients...')
    clients = []
    for i in range(n_client):
        train_loader, test_loader = data_loader.get_loader(i, use_global_test=True)
        clients.append(client(
            unlearn_mode=unlearn_mode,
            rank=i, 
            data_loader=(train_loader, test_loader),  # 每个客户端使用自己的数据（通过client_rank指定）
            device=device,
            encryption_key=ENCRYPTION_KEY,  # 传入密钥
            n_class=n_class,  # 传入类别数
            logger=logger,  # 传入日志器
            compression_mode=compression_mode,  # 传入压缩模式
            in_dim=in_dim,  # 传入输入通道数
            model_type=model_type  # 传入模型类型
        ))
    
    # 初始化遗忘学习模块
    unlearning_module = UnlearningModule(n_client, n_server1, client_assignments, logger=logger, compression_mode=compression_mode)
    
    # 根据配置选择要遗忘的客户端
    unlearning_clients = unlearning_module.select_unlearning_clients(
        unlearning_config, 
        seed=42  # 设置随机种子以确保可重复性（可选）
    )
    
    # 初始化外部攻击者
    external_attacker = None
    if enable_external_attacker:
        logger.info('\nInitialize External Attacker...')
        # 需要模型用于梯度反转攻击
        from model.lenet import lenet5, resnet18, cifar10cnn_v2
        if model_type == 'resnet18':
            attack_model = resnet18(n_class=n_class, in_dim=in_dim).to(device)
        elif model_type == 'cifar10cnn_v2':
            attack_model = cifar10cnn_v2(n_class=n_class, in_dim=in_dim).to(device)
        else:
            attack_model = lenet5(n_class=n_class, in_dim=in_dim).to(device)
        # 加载当前全局模型（如果存在）
        if os.path.exists('./cache/global_model_state.pkl'):
            try:
                attack_model.load_state_dict(torch.load('./cache/global_model_state.pkl', map_location=device))
            except:
                pass
        
        external_attacker = ExternalAttacker(
            attacker_id='external_attacker', 
            logger=logger,
            model=attack_model,
            n_class=n_class,
            device=device,
            data_loader=data_loader,
            save_visualization_epochs=save_visualization_epochs,
            save_iteration_steps=save_iteration_steps
        )
        logger.info('外部攻击者已初始化，将捕获客户端→中间服务器和中间服务器→中心服务器的梯度')

    # federated learning
    for e in range(n_epoch):
        warmup_round = (e == 0) 
        logger.info(f"当前压缩模式: {compression_mode}")
        logger.info(f"客户端加密密钥: {'已设置' if ENCRYPTION_KEY else '未设置'}")
    
        if compression_mode == 'none':
            logger.info("注意: compression_mode='none'，梯度将不进行稀疏化和加密")
        else:
            logger.info(f"使用压缩模式: {compression_mode}")
            logger.log_epoch_start(e + 1, n_epoch)
        
        # 获取已遗忘的客户端列表
        forgotten_clients = unlearning_module.get_forgotten_clients()
        if len(forgotten_clients) > 0:
            logger.info(f'[Info] 已遗忘客户端（不参与训练）: {sorted(forgotten_clients)}')
        
        is_first_round = (e == 0)
        if is_first_round and not compress_first_round:
            logger.info('[Info] 第一轮训练：不进行梯度压缩（保留完整梯度）')
        # Step 1: 所有客户端进行本地训练（排除已遗忘的客户端）
        logger.info('Step 1: Clients training...')
        active_count = 0
        # 存储原始数据用于攻击对比（仅第一轮，且不压缩时）
        original_data_dict = {}
        
        for i, c in enumerate(clients):
            if i not in forgotten_clients:
                # c.run(enable_compression=not warmup_round,enable_compression_all=False if compression_mode == 'none' else True)
                c.run(enable_compression=not warmup_round, enable_compression_all=False if compression_mode == 'none' else True, use_fixed_batch=use_fixed_batch)
                
                active_count += 1
                
                # 获取原始数据用于攻击对比（仅第一轮且不压缩时，用于评估攻击效果）
                if external_attacker is not None and (e == 0 and not compress_first_round):
                    try:
                        # 从客户端数据加载器获取一个batch的数据
                        train_loader = c.train_loader
                        for batch_data, batch_labels in train_loader:
                            # original_data_dict[i] = (batch_data[:1].to(device), batch_labels[:1].to(device))  # 只取第一个样本
                            original_data_dict[i] = (batch_data.to(device), batch_labels.to(device))  # 保存整个 batch（与 fixed batch 攻击一致）
                            
                            break
                    except Exception as ex:
                        if logger:
                            logger.debug(f'获取客户端{i}原始数据失败: {ex}')
                
                # 外部攻击者捕获客户端发送给中间服务器的梯度
                if external_attacker is not None:
                    try:
                        client_grads = torch.load(f'./cache/grads_{i}.pkl', map_location='cpu')
                        if logger:
                            logger.info(f'外部攻击者捕获客户端{i}梯度成功')
                        # 如果是加密压缩的梯度，需要解密（外部攻击者可能无法完全解密，但可以尝试）
                        # 这里假设外部攻击者只能看到加密压缩的梯度
                        encrypted_grads = client_grads.get('named_grads', {})
                        external_attacker.capture_client_to_server1_gradients(
                            epoch=e + 1,
                            client_rank=i,
                            gradients=encrypted_grads
                        )
                        
                    except Exception as ex:
                        if logger:
                            logger.warning(f'外部攻击者捕获客户端{i}梯度失败: {ex}')
            else:
                logger.info(f'  [Client {i}] 已遗忘，跳过训练')
        logger.info(f'  活跃客户端数: {active_count} / {len(clients)}')
        
        # Step 2: 中间服务器聚合各自区域的客户端梯度（排除已遗忘的客户端）
        logger.info('Step 2: Intermediate servers aggregating...')
        for s1 in server1s:
            s1.aggregate(
                forgotten_clients=forgotten_clients,
                epoch=e + 1,
                attack_rounds=attack_rounds
            )
            
            # 外部攻击者捕获中间服务器发送给中心服务器的梯度
            if external_attacker is not None:
                try:
                    server1_grads = torch.load(f'./cache/grads_agg1_{s1.rank}.pkl', map_location='cpu')
                    external_attacker.capture_server1_to_center_gradients(
                        epoch=e + 1,
                        server1_rank=s1.rank,
                        gradients=server1_grads.get('named_grads', {})
                    )
                except Exception as ex:
                    if logger:
                        logger.warning(f'外部攻击者捕获中间服务器{s1.rank}梯度失败: {ex}')
        
        # Step 3: 中心服务器聚合所有中间服务器的梯度并更新模型
        logger.info('Step 3: Center server aggregating...')
        center_server.aggregate(epoch=e + 1, attack_rounds=attack_rounds)
        
        # 验证配置一致性
        print(f"客户端压缩模式: {compression_mode}")
        print(f"服务器压缩模式: {compression_mode}")
        print(f"加密密钥: {'已设置' if ENCRYPTION_KEY else '未设置'}")
        assert clients[0].compressor.compression_mode == center_server.compressor.compression_mode


        # 外部攻击者对捕获的梯度进行攻击
        if external_attacker is not None:
            should_attack = (attack_rounds is None) or ((e + 1) in attack_rounds)
            if should_attack:
                logger.info('Step 3.5: External attacker performing gradient inversion attacks...')
                # 注意：外部攻击者获取的是加密压缩的梯度，可能无法完全重建
                # 但可以尝试攻击（结果可能不准确）
                attack_results = external_attacker.attack_captured_gradients(
                    epoch=e + 1,
                    attack_types=attack_types,
                    original_data_dict=original_data_dict if (e == 0 and not compress_first_round) else None
                )
                if logger:
                    # 处理嵌套的攻击结果
                    for source, source_results in attack_results.items():
                        logger.log_attack('external', f'external_attacker_{source}', e + 1, source_results)
        
        # 检查是否需要进行遗忘学习
        if unlearning_round > 0 and (e + 1) == unlearning_round:
            # 执行遗忘学习
            unlearning_module.perform_unlearning(
                clients=clients,
                server1s=server1s,
                center_server=center_server,
                unlearning_clients=unlearning_clients,
                unlearning_epochs=unlearning_epochs
            )
            # 更新已遗忘客户端列表（遗忘学习后）
            forgotten_clients = unlearning_module.get_forgotten_clients()
        
        # 记录epoch结束信息
        if len(center_server.accuracy) > 0:
            logger.log_epoch_end(e + 1, center_server.accuracy[-1])

    logger.info('\n训练完成！')
    # plot
    plot()


if __name__ == '__main__':
    federated_learning()
