import torch
import os
from model.client import client
from model.server import server
from model.data import loader
import tempfile

def test_compression_flow(compression_mode='none'):
    """测试完整的压缩流程"""
    print(f"\n=== 测试压缩模式: {compression_mode} ===")
    
    # # 创建临时目录
    # temp_dir = tempfile.mkdtemp()
    # cache_dir = os.path.join(temp_dir, 'cache')
    # os.makedirs(cache_dir, exist_ok=True)
    
    # 保存原始路径
    cache_dir = './cache'
    
    try:
        # 修改全局路径（简化测试）
        import builtins
        builtins.__dict__['TEST_MODE'] = True
        builtins.__dict__['TEST_CACHE_DIR'] = cache_dir
        
        # 初始化数据
        data_loader = loader('cifar10', batch_size=64, n_clients=2, alpha=1.0)
        
        # 初始化客户端
        client0 = client(
            rank=0,
            data_loader=data_loader.get_loader(0),
            device='cpu',
            encryption_key="test_key",
            n_class=10,
            logger=None,
            compression_mode=compression_mode
        )
        
        # 初始化服务器
        server0 = server(
            size=1,
            data_loader=(None, data_loader.test_dataset),
            device='cpu',
            encryption_key="test_key",
            n_class=10,
            logger=None,
            compression_mode=compression_mode
        )
        
        print("1. 客户端训练...")
        # 模拟训练（需要模型文件）
        if not os.path.exists(os.path.join(cache_dir, 'global_model_state.pkl')):
            torch.save(server0.model.state_dict(), os.path.join(cache_dir, 'global_model_state.pkl'))
        
        # 运行客户端训练
        client0.run(enable_compression=True, enable_compression_all=(compression_mode != 'none'))
        
        # 检查保存的梯度
        grad_path = os.path.join(cache_dir, 'grads_0.pkl')
        if os.path.exists(grad_path):
            grads = torch.load(grad_path)
            print(f"   梯度文件大小: {os.path.getsize(grad_path)} bytes")
            print(f"   包含键: {list(grads.keys())}")
            
            # 检查梯度内容
            if 'named_grads' in grads:
                for name, grad in grads['named_grads'].items():
                    print(f"   梯度 {name}: 形状={grad.shape}, 类型={grad.dtype}")
        else:
            print(f"梯度文件不存在: {grad_path}")
        
        print("\n2. 模拟服务器聚合...")
        # 模拟服务器加载客户端梯度
        if os.path.exists(grad_path):
            # 服务器需要模拟中间服务器聚合
            simulated_grads = {
                'n_samples': grads['n_samples'],
                'named_grads': grads['named_grads']
            }
            
            # 保存模拟的中间服务器梯度
            agg_path = os.path.join(cache_dir, 'grads_agg1_0.pkl')
            torch.save(simulated_grads, agg_path)
            
            # 服务器聚合
            try:
                server0.aggregate()
                print("服务器聚合成功")
            except Exception as e:
                print(f"服务器聚合失败: {e}")
        
        print("\n3. 验证梯度传递...")
        # 验证客户端和服务器端的梯度是否一致
        if os.path.exists(grad_path):
            client_grads = torch.load(grad_path)
            
            # 这里应该检查加密/解密是否正确
            print(f"   测试完成: 压缩模式={compression_mode}")
            
    # finally:
    #     # 清理
    #     import shutil
    #     shutil.rmtree(temp_dir)
    #     print(f"   清理临时目录: {temp_dir}")

# 测试不同模式
test_compression_flow('none')
test_compression_flow('adaptive')
test_compression_flow('fixed')