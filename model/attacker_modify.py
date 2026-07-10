import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict
from torch.autograd import Variable
import torch.nn.functional as F


class GradientInversionAttacker:
    """梯度反转攻击者
    
    实现梯度反转攻击（Gradient Inversion Attack）和梯度深度泄露攻击（Deep Leakage from Gradients）
    从梯度重建训练数据和标签
    """
    
    def __init__(self, attacker_type: str, attacker_id: str = None, logger=None, 
                 model=None, n_class=10, device=None, data_loader=None,
                 save_visualization_epochs=None, save_iteration_steps=None):
        """
        Args:
            attacker_type: 攻击者类型 ('external', 'intermediate_server', 'center_server')
            attacker_id: 攻击者标识符
            logger: 日志记录器
            model: 模型实例（用于梯度反转）
            n_class: 类别数
            device: 设备
            data_loader: 数据加载器（用于获取原始数据对比）
            save_visualization_epochs: 要保存可视化结果的轮次列表，None表示所有轮次都保存
                                       例如 [1, 3, 5] 表示只在第1,3,5轮保存对比图
            save_iteration_steps: 在优化迭代的哪些步数保存重建结果图片，None表示不保存中间结果
                                  例如 [100, 500, 1000] 表示在第100,500,1000次迭代时保存
        """
        self.attacker_type = attacker_type
        self.attacker_id = attacker_id
        self.logger = logger
        self.model = model
        self.n_class = n_class
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data_loader = data_loader
        self.attack_history = []
        self.save_visualization_epochs = save_visualization_epochs  # 保存可视化结果的轮次列表
        self.save_iteration_steps = save_iteration_steps  # 保存优化迭代中间结果的步数列表
        self.saved_original_images = set()  # 跟踪已保存的原始图像，格式: (epoch, identifier, attack_type)
        os.makedirs('./cache/attack_results', exist_ok=True)
    
    def _infer_input_shape(self, gradients: Dict, batch_size: int):
        """从梯度推断输入形状"""
        input_shape = None
        
        for name, grad in gradients.items():
            name_low = name.lower()
            if ('conv' in name_low and 'weight' in name_low) or 'conv1' in name_low:
                if len(grad.shape) >= 4:  # Conv2d weight: [out_channels, in_channels, H, W]
                    in_channels = grad.shape[1]
                    
                    # 根据数据集类型确定输入大小
                    if self.data_loader:
                        data_name = str(self.data_loader).lower()
                        if 'cifar' in data_name:
                            # CIFAR: 32x32
                            input_shape = (batch_size, in_channels, 32, 32)
                        elif 'mnist' in data_name:
                            # MNIST: 28x28 或 32x32（取决于模型）
                            # 检查模型结构以确定
                            if self.model is not None:
                                # 尝试推断
                                try:
                                    # 创建一个测试输入看看维度是否匹配
                                    test_input = torch.randn(1, in_channels, 28, 28).to(self.device)
                                    with torch.no_grad():
                                        output = self.model(test_input)
                                    # 如果成功，使用28x28
                                    input_shape = (batch_size, in_channels, 28, 28)
                                except:
                                    # 否则使用32x32
                                    input_shape = (batch_size, in_channels, 32, 32)
                            else:
                                input_shape = (batch_size, in_channels, 28, 28)
                        elif 'imagenet' in data_name:
                            # ImageNet: 224x224
                            input_shape = (batch_size, in_channels, 224, 224)
                        else:
                            # 默认使用32x32
                            input_shape = (batch_size, in_channels, 32, 32)
                    else:
                        # 默认使用32x32（CIFAR标准）
                        input_shape = (batch_size, in_channels, 32, 32)
                    break
        
        if input_shape is None:
            # 默认使用CIFAR形状（32x32）
            input_shape = (batch_size, 3, 32, 32)
            
        return input_shape
    
    def _match_param_names(self, gradients: Dict):
        """匹配梯度键名与模型参数名"""
        model_param_names = {name for name, _ in self.model.named_parameters()}
        grad_names = set(gradients.keys())
        
        # 完全匹配
        matching_names = model_param_names & grad_names
        
        if len(matching_names) == 0:
            # 尝试部分匹配
            for model_name in model_param_names:
                for grad_name in grad_names:
                    # 检查是否模型名包含梯度名或反之
                    if model_name in grad_name or grad_name in model_name:
                        matching_names.add(model_name)
                        break
            
            if len(matching_names) == 0 and self.logger:
                self.logger.warning(f'梯度键名与模型参数名不匹配。模型参数: {list(model_param_names)[:3]}..., 梯度键: {list(grad_names)[:3]}...')
        
        return matching_names
    
    def _compute_gradient_loss(self, dummy_data, dummy_labels, gradients):
        """计算梯度匹配损失（使用高阶梯度）"""
        # 清除模型参数的梯度
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.zero_()
        
        # 前向传播
        output = self.model(dummy_data)
        loss = F.cross_entropy(output, dummy_labels)
        
        # 获取模型参数列表
        model_params = list(self.model.parameters())
        
        # 计算梯度（创建计算图以进行二阶梯度）
        grads = torch.autograd.grad(loss, model_params, create_graph=True, allow_unused=True)
        
        # 计算梯度匹配损失
        grad_loss = 0.0
        count = 0
        
        for i, (name, param) in enumerate(self.model.named_parameters()):
            if name in gradients and grads[i] is not None:
                true_grad = gradients[name]
                if isinstance(true_grad, torch.Tensor):
                    true_grad = true_grad.to(self.device)
                    
                    # 确保形状匹配
                    if grads[i].shape == true_grad.shape:
                        # 计算MSE损失
                        mse_loss = F.mse_loss(grads[i], true_grad)
                        
                        # 计算余弦相似度
                        cos_sim = F.cosine_similarity(
                            grads[i].flatten(), 
                            true_grad.flatten(), 
                            dim=0
                        )
                        
                        # 组合损失：MSE + (1 - 余弦相似度)
                        grad_loss += mse_loss + (1 - cos_sim)
                        count += 1
        
        if count == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        return grad_loss / count
    
    def _postprocess_image_for_vis(self, x: torch.Tensor) -> torch.Tensor:
        """将模型输入数据转换到可视范围[0,1]"""
        if x is None:
            return None
        
        x = x.detach().cpu()
        
        # 获取通道数
        if x.dim() < 4:
            return x
        num_channels = x.size(1)
        
        # 常见数据集的标准化参数
        if num_channels == 1:
            # MNIST: 单通道
            mnist_mean = torch.tensor([0.1307]).view(1, 1, 1, 1)
            mnist_std = torch.tensor([0.3081]).view(1, 1, 1, 1)
            # 根据数据范围判断是否已经标准化
            if x.min() < -0.5 or x.max() > 1.5:
                # 尝试反标准化（假设是MNIST）
                x = x * mnist_std + mnist_mean
        else:
            # CIFAR: 三通道
            cifar_mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
            cifar_std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1)
            # 根据数据范围判断是否已经标准化
            if x.min() < -1.0 or x.max() > 2.0:
                # 尝试反标准化（假设是CIFAR）
                x = x * cifar_std + cifar_mean
        
        # 最终限制到[0,1]
        x = torch.clamp(x, 0.0, 1.0)
        return x
    
    def perform_gradient_inversion(self, gradients: Dict, epoch: int, client_rank: int = None,
                                  batch_size: int = 1, num_iterations: int = 1000,
                                  lr: float = 0.1, original_data: torch.Tensor = None,
                                  original_labels: torch.Tensor = None) -> Dict:
        """执行梯度反转攻击（改进版）
        
        假设攻击者知道标签，优化dummy_data使得模型的参数梯度与true_gradients匹配
        
        Args:
            gradients: 梯度字典 {layer_name: gradient_tensor}
            epoch: 当前训练轮次
            client_rank: 客户端rank（如果已知）
            batch_size: 重建的批次大小
            num_iterations: 优化迭代次数
            lr: 学习率
            original_data: 原始数据（用于对比，可选）
            original_labels: 原始标签（用于对比，可选）
            
        Returns:
            attack_result: 攻击结果字典
        """
        attack_result = {
            'attack_type': 'gradient_inversion',
            'attacker_type': self.attacker_type,
            'attacker_id': self.attacker_id,
            'epoch': epoch,
            'client_rank': client_rank,
            'success': False,
            'label_accuracy': 0.0,
            'data_similarity': 0.0,
            'details': {}
        }
        self.logger.info("开始进行修改后的！！！！梯度反演攻击")
        try:
            if self.model is None:
                raise ValueError("模型未提供，无法进行梯度反转攻击")
            
            # 将模型设置为训练模式并移动到设备
            self.model.train()
            self.model.to(self.device)
            
            # 推断输入形状
            input_shape = self._infer_input_shape(gradients, batch_size)
            
            # 预处理原始数据（如果有）
            if original_data is not None:
                original_data = original_data.to(self.device)
                original_data_processed = original_data
            else:
                original_data_processed = None
            
            # 获取已知标签（如果提供了）
            known_labels = None
            if original_labels is not None:
                known_labels = original_labels.to(self.device)
            else:
                # 如果没有提供标签，我们无法进行梯度反转攻击（因为需要已知标签）
                if self.logger:
                    self.logger.warning("梯度反转攻击需要已知标签，但未提供original_labels")
                attack_result['details']['error'] = '梯度反转攻击需要已知标签'
                return attack_result
            
            # 初始化dummy数据（使用更好的初始化）
            dummy_data = torch.randn(input_shape, device=self.device) * 0.1
            dummy_data = nn.Parameter(dummy_data)
            
            # 优化器
            optimizer = optim.Adam([dummy_data], lr=lr)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iterations)
            
            best_loss = float('inf')
            best_data = None
            best_iteration = 0
            
            # 优化循环
            for iteration in range(num_iterations):
                optimizer.zero_grad()
                
                # 计算梯度匹配损失（使用高阶梯度）
                grad_loss = self._compute_gradient_loss(dummy_data, known_labels, gradients)
                
                # 如果梯度损失太大，跳过此次迭代
                if grad_loss.item() > 1e5:
                    if iteration == 0 and self.logger:
                        self.logger.debug(f'梯度损失过大: {grad_loss.item():.6f}')
                    continue
                
                # 计算总变分损失（图像平滑）
                tv_loss = self._total_variation_loss(dummy_data)
                
                # 总损失
                total_loss = grad_loss + 0.01 * tv_loss
                
                # 反向传播
                total_loss.backward()
                optimizer.step()
                scheduler.step()
                
                # 限制数据范围
                with torch.no_grad():
                    dummy_data.data = torch.clamp(dummy_data, -2.5, 2.5)
                
                # 记录最佳结果
                current_loss = total_loss.item()
                if current_loss < best_loss:
                    best_loss = current_loss
                    best_data = dummy_data.detach().clone()
                    best_iteration = iteration
                
                # 检查是否需要在当前迭代步数保存中间结果
                current_iteration = iteration + 1
                should_save_iteration = (self.save_iteration_steps is not None) and (current_iteration in self.save_iteration_steps)
                if should_save_iteration:
                    # 保存当前迭代的重建结果（只保存重建图像）
                    current_data = dummy_data.detach().clone()
                    try:
                        # 使用原始数据（如果有）作为ground truth，但只保存重建图像
                        vis_original_data = original_data_processed if original_data_processed is not None else current_data
                        vis_original_labels = known_labels if known_labels is not None else known_labels
                        # 在攻击方法中添加
                        if self.logger:
                            self.logger.debug(f"original_data 原始形状: {original_data.shape if original_data is not None else 'None'}")
                            self.logger.debug(f"original_data 数值范围: [{original_data.min() if original_data is not None else 'N/A'}, "
                        f"{original_data.max() if original_data is not None else 'N/A'}]")
                        self._visualize_reconstruction(
                            vis_original_data, current_data, vis_original_labels, known_labels,
                            epoch, client_rank, 'gradient_inversion', iteration_step=current_iteration,
                            save_reconstruction_only=True
                        )
                    except Exception as e:
                        if self.logger:
                            self.logger.warning(f'保存迭代{current_iteration}的重建结果失败: {e}')
                
                # 每100次迭代输出一次
                if (iteration + 1) % 100 == 0 and self.logger:
                    self.logger.debug(f'  迭代 {iteration+1}/{num_iterations}, Loss: {current_loss:.6f}, '
                                    f'Grad Loss: {grad_loss.item():.6f}, Best Loss: {best_loss:.6f}')
            
            # 检查是否有有效的结果
            if best_data is None:
                # 所有迭代都失败了，使用最终值
                if self.logger:
                    self.logger.warning('所有迭代都失败，使用最终值作为重建结果')
                best_data = dummy_data.detach().clone()
            
            # 重建的数据
            reconstructed_data = self._postprocess_image_for_vis(best_data)
            
            # 计算标签准确率（总是100%，因为我们使用了已知标签）
            label_accuracy = 1.0
            
            # 计算数据相似度（如果有原始数据）
            data_similarity = 0.0
            if original_data_processed is not None and reconstructed_data is not None:
                try:
                    # 计算结构相似性
                    original_norm = original_data_processed.view(original_data_processed.size(0), -1)
                    recon_norm = reconstructed_data.view(reconstructed_data.size(0), -1)
                    
                    # 计算余弦相似度
                    cos_sim = F.cosine_similarity(original_norm, recon_norm, dim=1)
                    data_similarity = cos_sim.mean().item()
                except:
                    data_similarity = 0.0
            
            attack_result['success'] = True
            attack_result['label_accuracy'] = label_accuracy
            attack_result['data_similarity'] = data_similarity
            attack_result['details'] = {
                'reconstructed_data_shape': list(reconstructed_data.shape) if reconstructed_data is not None else None,
                'reconstructed_labels': known_labels.cpu().tolist() if known_labels is not None else None,
                'best_loss': float(best_loss) if best_loss != float('inf') else None,
                'best_iteration': best_iteration,
                'num_iterations': num_iterations,
                'all_iterations_failed': (best_data is None)
            }
            
            # 保存重建结果
            if reconstructed_data is not None and known_labels is not None:
                identifier = self._get_identifier(client_rank)
                save_path = f'./cache/attack_results/gradient_inversion_epoch{epoch}_{identifier}.pkl'
                try:
                    torch.save({
                        'reconstructed_data': reconstructed_data.cpu(),
                        'reconstructed_labels': known_labels.cpu(),
                        'original_data': original_data.cpu() if original_data is not None else None,
                        'original_labels': known_labels.cpu() if known_labels is not None else None,
                        'epoch': epoch,
                        'client_rank': client_rank,
                        'attacker_id': self.attacker_id,
                        'label_accuracy': label_accuracy,
                        'data_similarity': data_similarity
                    }, save_path)
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f'保存重建结果失败: {e}')
                
                # 可视化：保存地面真实图像（如果还没有）和最终重建图像
                should_save = (self.save_visualization_epochs is None) or (epoch in self.save_visualization_epochs)
                if should_save:
                    try:
                        vis_original_data = original_data_processed if original_data_processed is not None else reconstructed_data
                        vis_original_labels = known_labels if known_labels is not None else known_labels
                        # 保存地面真实图像和最终重建图像
                        self._visualize_reconstruction(
                            vis_original_data, reconstructed_data, vis_original_labels, known_labels,
                            epoch, client_rank, 'gradient_inversion', iteration_step=None,
                            save_reconstruction_only=False
                        )
                    except Exception as e:
                        if self.logger:
                            self.logger.warning(f'可视化失败: {e}')
            
        except Exception as e:
            attack_result['details']['error'] = str(e)
            if self.logger:
                self.logger.error(f'梯度反转攻击失败: {e}')
        
        self.attack_history.append(attack_result)
        return attack_result
    
    def perform_deep_leakage(self, gradients: Dict, epoch: int, client_rank: int = None,
                            batch_size: int = 1, num_iterations: int = 2000,
                            lr: float = 0.1, original_data: torch.Tensor = None,
                            original_labels: torch.Tensor = None) -> Dict:
        """执行梯度深度泄露攻击（Deep Leakage from Gradients）（改进版）
        
        使用DLG方法从梯度重建训练数据，同时优化数据和标签
        
        Args:
            gradients: 梯度字典
            epoch: 当前训练轮次
            client_rank: 客户端rank
            batch_size: 重建的批次大小
            num_iterations: 优化迭代次数
            lr: 学习率
            original_data: 原始数据（用于对比）
            original_labels: 原始标签（用于对比）
            
        Returns:
            attack_result: 攻击结果字典
        """
        attack_result = {
            'attack_type': 'deep_leakage',
            'attacker_type': self.attacker_type,
            'attacker_id': self.attacker_id,
            'epoch': epoch,
            'client_rank': client_rank,
            'success': False,
            'label_accuracy': 0.0,
            'data_similarity': 0.0,
            'details': {}
        }
        self.logger.info("开始进行修改后的！！！！深度泄露攻击")
        try:
            if self.model is None:
                raise ValueError("模型未提供，无法进行深度泄露攻击")
            
            # 将模型设置为训练模式并移动到设备
            self.model.train()
            self.model.to(self.device)
            
            # 推断输入形状
            input_shape = self._infer_input_shape(gradients, batch_size)
            
            # 预处理原始数据（如果有）
            if original_data is not None:
                original_data = original_data.to(self.device)
                original_data_processed = original_data
            else:
                original_data_processed = None
            
            # 初始化dummy数据（使用更好的初始化）
            dummy_data = torch.randn(input_shape, device=self.device) * 0.1
            dummy_data = nn.Parameter(dummy_data)
            
            # 初始化dummy标签（使用softmax logits）
            dummy_logits = torch.randn(batch_size, self.n_class, device=self.device) * 0.01
            dummy_logits = nn.Parameter(dummy_logits)
            
            # 优化器（使用不同的学习率）
            optimizer = optim.Adam([
                {'params': [dummy_data], 'lr': lr},
                {'params': [dummy_logits], 'lr': lr * 0.5}
            ])
            
            # 学习率调度器
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iterations)
            
            best_loss = float('inf')
            best_data = None
            best_labels = None
            best_iteration = 0
            
            # 优化循环
            for iteration in range(num_iterations):
                optimizer.zero_grad()
                
                # 从logits获取标签（softmax + argmax）
                dummy_probs = F.softmax(dummy_logits, dim=1)
                dummy_labels = dummy_probs.argmax(dim=1)
                
                # 计算梯度匹配损失（使用高阶梯度）
                grad_loss = self._compute_gradient_loss(dummy_data, dummy_labels, gradients)
                
                # 如果梯度损失太大，跳过此次迭代
                if grad_loss.item() > 1e5:
                    if iteration == 0 and self.logger:
                        self.logger.debug(f'梯度损失过大: {grad_loss.item():.6f}')
                    continue
                
                # 计算总变分损失（图像平滑）
                tv_loss = self._total_variation_loss(dummy_data)
                
                # 标签熵损失（鼓励one-hot标签）
                label_entropy = -torch.sum(dummy_probs * torch.log(dummy_probs + 1e-10), dim=1).mean()
                
                # 总损失（根据DLG论文）
                total_loss = grad_loss + 0.01 * tv_loss + 0.001 * label_entropy
                
                # 反向传播
                total_loss.backward()
                optimizer.step()
                scheduler.step()
                
                # 限制数据范围
                with torch.no_grad():
                    dummy_data.data = torch.clamp(dummy_data, -2.5, 2.5)
                
                # 记录最佳结果
                current_loss = total_loss.item()
                if current_loss < best_loss:
                    best_loss = current_loss
                    best_data = dummy_data.detach().clone()
                    best_labels = dummy_labels.detach().clone()
                    best_iteration = iteration
                
                # 检查是否需要在当前迭代步数保存中间结果
                current_iteration = iteration + 1
                should_save_iteration = (self.save_iteration_steps is not None) and (current_iteration in self.save_iteration_steps)
                if should_save_iteration:
                    # 保存当前迭代的重建结果（只保存重建图像）
                    current_data = dummy_data.detach().clone()
                    current_labels = dummy_labels.detach().clone()
                    try:
                        # 使用原始数据（如果有）作为ground truth，但只保存重建图像
                        vis_original_data = original_data_processed if original_data_processed is not None else current_data
                        vis_original_labels = original_labels if original_labels is not None else current_labels
                        self._visualize_reconstruction(
                            vis_original_data, current_data, vis_original_labels, current_labels,
                            epoch, client_rank, 'deep_leakage', iteration_step=current_iteration,
                            save_reconstruction_only=True
                        )
                    except Exception as e:
                        if self.logger:
                            self.logger.warning(f'保存迭代{current_iteration}的重建结果失败: {e}')
                
                if (iteration + 1) % 200 == 0 and self.logger:
                    self.logger.debug(f'  迭代 {iteration+1}/{num_iterations}, Loss: {current_loss:.6f}, '
                                    f'Grad Loss: {grad_loss.item():.6f}, Best Loss: {best_loss:.6f}')
            
            # 检查是否有有效的结果
            if best_data is None or best_labels is None:
                if self.logger:
                    self.logger.warning('所有迭代都失败，使用最终值作为重建结果')
                best_data = dummy_data.detach().clone()
                best_labels = dummy_labels.detach().clone()
            
            reconstructed_data = self._postprocess_image_for_vis(best_data)
            reconstructed_labels = best_labels
            
            # 计算标签准确率（如果有原始标签）
            label_accuracy = 0.0
            if original_labels is not None and reconstructed_labels is not None:
                try:
                    original_labels_tensor = original_labels.to(self.device)
                    label_accuracy = (reconstructed_labels == original_labels_tensor).float().mean().item()
                except:
                    label_accuracy = 0.0
            
            # 计算数据相似度
            data_similarity = 0.0
            if original_data_processed is not None and reconstructed_data is not None:
                try:
                    original_norm = original_data_processed.view(original_data_processed.size(0), -1)
                    recon_norm = reconstructed_data.view(reconstructed_data.size(0), -1)
                    cos_sim = F.cosine_similarity(original_norm, recon_norm, dim=1)
                    data_similarity = cos_sim.mean().item()
                except:
                    data_similarity = 0.0
            
            attack_result['success'] = True
            attack_result['label_accuracy'] = label_accuracy
            attack_result['data_similarity'] = data_similarity
            attack_result['details'] = {
                'reconstructed_data_shape': list(reconstructed_data.shape) if reconstructed_data is not None else None,
                'reconstructed_labels': reconstructed_labels.cpu().tolist() if reconstructed_labels is not None else None,
                'best_loss': float(best_loss) if best_loss != float('inf') else None,
                'best_iteration': best_iteration,
                'num_iterations': num_iterations,
                'all_iterations_failed': (best_data is None or best_labels is None)
            }
            
            # 保存重建结果
            if reconstructed_data is not None and reconstructed_labels is not None:
                identifier = self._get_identifier(client_rank)
                save_path = f'./cache/attack_results/deep_leakage_epoch{epoch}_{identifier}.pkl'
                try:
                    torch.save({
                        'reconstructed_data': reconstructed_data.cpu(),
                        'reconstructed_labels': reconstructed_labels.cpu(),
                        'original_data': original_data.cpu() if original_data is not None else None,
                        'original_labels': original_labels.cpu() if original_labels is not None else None,
                        'epoch': epoch,
                        'client_rank': client_rank,
                        'attacker_id': self.attacker_id,
                        'label_accuracy': label_accuracy,
                        'data_similarity': data_similarity
                    }, save_path)
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f'保存重建结果失败: {e}')
                
                # 可视化：保存地面真实图像（如果还没有）和最终重建图像
                should_save = (self.save_visualization_epochs is None) or (epoch in self.save_visualization_epochs)
                if should_save:
                    try:
                        vis_original_data = original_data_processed if original_data_processed is not None else reconstructed_data
                        vis_original_labels = original_labels if original_labels is not None else reconstructed_labels
                        # 保存地面真实图像和最终重建图像
                        self._visualize_reconstruction(
                            vis_original_data, reconstructed_data, vis_original_labels, reconstructed_labels,
                            epoch, client_rank, 'deep_leakage', iteration_step=None,
                            save_reconstruction_only=False
                        )
                    except Exception as e:
                        if self.logger:
                            self.logger.warning(f'可视化失败: {e}')
            
        except Exception as e:
            attack_result['details']['error'] = str(e)
            if self.logger:
                self.logger.error(f'深度泄露攻击失败: {e}')
        
        self.attack_history.append(attack_result)
        return attack_result
    
    def _get_identifier(self, client_rank: int = None) -> str:
        """根据攻击者类型和client_rank生成标识符"""
        if client_rank is not None:
            # 外部攻击者或中间服务器攻击特定客户端时，使用client标识
            return f'client{client_rank}'
        else:
            # 内部攻击者（中心服务器或中间服务器聚合梯度）使用attacker_id
            return self.attacker_id if self.attacker_id else 'unknown'
    
    def _total_variation_loss(self, x):
        """总变分损失（用于图像平滑）"""
        batch_size = x.size(0)
        h_x = x.size(2)
        w_x = x.size(3)
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x-1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x-1]), 2).sum()
        return 2 * (h_tv / count_h + w_tv / count_w) / batch_size
    
    def _tensor_size(self, t):
        return t.size(1) * t.size(2) * t.size(3)
    
    def _visualize_reconstruction(self, original_data: torch.Tensor, reconstructed_data: torch.Tensor,
                                 original_labels: torch.Tensor, reconstructed_labels: torch.Tensor,
                                 epoch: int, client_rank: int, attack_type: str, iteration_step: int = None,
                                 save_reconstruction_only: bool = False):
        """可视化重建结果
        
        Args:
            original_data: 原始数据（地面真实图像）
            reconstructed_data: 重建数据
            original_labels: 原始标签
            reconstructed_labels: 重建标签
            epoch: 训练轮次
            client_rank: 客户端rank
            attack_type: 攻击类型
            iteration_step: 优化迭代步数（如果提供，会在文件名中包含此信息）
            save_reconstruction_only: 如果为True，只保存重建图像；如果为False，保存地面真实图像和重建图像
        """
        try:
            # 确保数据在可视范围内
            original_vis = self._postprocess_image_for_vis(original_data)
            reconstructed_vis = self._postprocess_image_for_vis(reconstructed_data)
            
            # 检查是否有真正的原始数据（通过比较数据是否相同来判断）
            # 如果原始数据和重建数据非常相似，说明没有真正的原始数据
            has_real_original = True
            try:
                # 计算两个张量的差异
                if original_vis.shape == reconstructed_vis.shape:
                    diff = torch.abs(original_vis - reconstructed_vis)
                    mean_diff = diff.mean().item()
                    # 如果平均差异很小（小于阈值），说明可能是同一数据
                    if mean_diff < 1e-3:
                        has_real_original = False
                else:
                    # 形状不同，肯定不是同一数据
                    has_real_original = True
            except:
                # 如果比较失败，假设有真正的原始数据
                has_real_original = True
            
            # 检查通道数，决定如何处理
            if original_vis.dim() >= 4:
                num_channels = original_vis.size(1)
            else:
                # 如果维度不对，尝试从原始数据推断
                if original_data.dim() >= 4:
                    num_channels = original_data.size(1)
                else:
                    num_channels = 3  # 默认假设是3通道
            
            # 转换为numpy，根据通道数处理
            if num_channels == 1:
                # MNIST单通道：需要squeeze掉通道维度，使用灰度显示
                original_vis_np = original_vis.squeeze(1).numpy()  # [batch, H, W]
                reconstructed_vis_np = reconstructed_vis.squeeze(1).numpy()  # [batch, H, W]
                cmap = 'gray'
            else:
                # CIFAR三通道：permute到 [batch, H, W, C]
                original_vis_np = original_vis.permute(0, 2, 3, 1).numpy()  # [batch, H, W, C]
                reconstructed_vis_np = reconstructed_vis.permute(0, 2, 3, 1).numpy()  # [batch, H, W, C]
                cmap = None
            
            # 确保标签是tensor
            if not isinstance(original_labels, torch.Tensor):
                original_labels = torch.tensor(original_labels)
            if not isinstance(reconstructed_labels, torch.Tensor):
                reconstructed_labels = torch.tensor(reconstructed_labels)

            # 生成标识符
            identifier = self._get_identifier(client_rank)
            
            num_samples = min(original_data.size(0), 8)  # 最多显示8个样本
            
            # 保存地面真实图像（只在第一次保存，且不是只保存重建图像时，且有真正的原始数据时）
            original_image_key = (epoch, identifier, attack_type)
            should_save_ground_truth = (original_image_key not in self.saved_original_images and 
                                       not save_reconstruction_only and 
                                       iteration_step is None and
                                       has_real_original)  # 只有在有真正的原始数据时才保存
            
            if should_save_ground_truth and num_samples > 0:
                # 保存地面真实图像
                fig_ground_truth,  axes = plt.subplots(2, num_samples, figsize=(2 * num_samples, 4))

                # 当 num_samples == 1 时，axes 不是二维数组，需要手动包一层
                if num_samples == 1:
                    axes = axes.reshape(2, 1)

                axes_gt = axes[0]
                axes_recon = axes[1]

                for i in range(num_samples):
                    axes_gt[i].imshow(original_vis_np[i], cmap=cmap)
                    axes_gt[i].set_title(f'Ground Truth\n标签: {original_labels[i].item()}')
                    axes_gt[i].axis('off')
                plt.tight_layout()
                ground_truth_save_path = f'./cache/attack_results/{attack_type}_epoch{epoch}_{identifier}_ground_truth.png'
                plt.savefig(ground_truth_save_path, dpi=150, bbox_inches='tight')
                plt.close(fig_ground_truth)
                # 标记为已保存
                self.saved_original_images.add(original_image_key)
                if self.logger:
                    self.logger.info(f'  地面真实图像已保存: {ground_truth_save_path}')
            
            # 保存重建图像
            if num_samples > 0:
                fig_reconstruction, axes_recon = plt.subplots(1, num_samples, figsize=(2*num_samples, 2))
                if num_samples == 1:
                    axes_recon = [axes_recon]
                for i in range(num_samples):
                    axes_gt[i].imshow(original_vis_np[i], cmap=cmap, interpolation='bicubic')
                    axes_recon[i].imshow(reconstructed_vis_np[i], cmap=cmap, interpolation='bicubic')

                    pred_label = reconstructed_labels[i].item()
                    if not save_reconstruction_only and has_real_original:
                        # 只有在有真正的原始数据时才显示匹配标记
                        true_label = original_labels[i].item()
                        match = "✓" if pred_label == true_label else "✗"
                        axes_recon[i].set_title(f'重建 {match}\n标签: {pred_label}')
                    else:
                        axes_recon[i].set_title(f'重建\n标签: {pred_label}')
                    axes_recon[i].axis('off')
                plt.tight_layout()
                iter_suffix = f'_iter{iteration_step}' if iteration_step is not None else ''
                reconstruction_save_path = f'./cache/attack_results/{attack_type}_epoch{epoch}_{identifier}{iter_suffix}_reconstruction.png'
                plt.savefig(reconstruction_save_path, dpi=150, bbox_inches='tight')
                plt.close(fig_reconstruction)
                
                if self.logger:
                    if iteration_step is not None:
                        self.logger.info(f'  迭代{iteration_step}重建图像已保存: {reconstruction_save_path}')
                    else:
                        self.logger.info(f'  重建图像已保存: {reconstruction_save_path}')
                
        except Exception as e:
            if self.logger:
                self.logger.warning(f'可视化失败: {e}')
            import traceback
            if self.logger:
                self.logger.debug(f'可视化失败详细错误: {traceback.format_exc()}')


class ExternalAttacker(GradientInversionAttacker):
    """外部攻击者"""
    
    def __init__(self, attacker_id: str = "external_attacker", logger=None, 
                 model=None, n_class=10, device=None, data_loader=None,
                 save_visualization_epochs=None, save_iteration_steps=None):
        super().__init__('external', attacker_id, logger, model, n_class, device, data_loader,
                        save_visualization_epochs=save_visualization_epochs,
                        save_iteration_steps=save_iteration_steps)
        self.captured_gradients = {}
    
    def capture_client_to_server1_gradients(self, epoch: int, client_rank: int, gradients: Dict):
        """捕获客户端发送给中间服务器的梯度"""
        if epoch not in self.captured_gradients:
            self.captured_gradients[epoch] = {}
        self.captured_gradients[epoch][f'client_{client_rank}'] = gradients
    
    def capture_server1_to_center_gradients(self, epoch: int, server1_rank: int, gradients: Dict):
        """捕获中间服务器发送给中心服务器的梯度"""
        if epoch not in self.captured_gradients:
            self.captured_gradients[epoch] = {}
        self.captured_gradients[epoch][f'server1_{server1_rank}'] = gradients
    
    def attack_captured_gradients(self, epoch: int, attack_types: List[str] = None,
                                 original_data_dict: Dict = None) -> Dict:
        """对捕获的梯度进行攻击
        
        Args:
            epoch: 训练轮次
            attack_types: 攻击类型列表
            original_data_dict: 原始数据字典 {client_rank: (data, labels)}
        """
        if attack_types is None:
            attack_types = ['gradient_inversion', 'deep_leakage']
        
        if epoch not in self.captured_gradients:
            return {}
        
        results = {}
        for source, gradients in self.captured_gradients[epoch].items():
            # 提取client_rank
            client_rank = None
            if source.startswith('client_'):
                client_rank = int(source.split('_')[1])
            
            # 获取原始数据（如果提供）
            original_data = None
            original_labels = None
            if original_data_dict is not None and client_rank is not None:
                if client_rank in original_data_dict:
                    original_data, original_labels = original_data_dict[client_rank]
            
            source_results = {}
            for attack_type in attack_types:
                if attack_type == 'gradient_inversion':
                    result = self.perform_gradient_inversion(
                        gradients, epoch, client_rank,
                        original_data=original_data, original_labels=original_labels
                    )
                    source_results[attack_type] = result
                elif attack_type == 'deep_leakage':
                    result = self.perform_deep_leakage(
                        gradients, epoch, client_rank,
                        original_data=original_data, original_labels=original_labels
                    )
                    source_results[attack_type] = result
            
            results[source] = source_results
        
        return results


class InternalAttacker(GradientInversionAttacker):
    """内部攻击者（诚实且好奇的服务器）"""
    
    def __init__(self, attacker_type: str, attacker_id: str, logger=None,
                 model=None, n_class=10, device=None, data_loader=None,
                 save_visualization_epochs=None, save_iteration_steps=None):
        super().__init__(attacker_type, attacker_id, logger, model, n_class, device, data_loader,
                        save_visualization_epochs=save_visualization_epochs,
                        save_iteration_steps=save_iteration_steps)
    
    


    def attack_on_gradients(self, gradients: Dict, epoch: int, client_rank: int = None,
                           attack_types: List[str] = None, original_data: torch.Tensor = None,
                           original_labels: torch.Tensor = None) -> Dict:
        """对接收到的梯度进行攻击"""
        

        if attack_types is None:
            attack_types = ['gradient_inversion', 'deep_leakage']
        
        results = {}
        for attack_type in attack_types:
            if attack_type == 'gradient_inversion':
                result = self.perform_gradient_inversion(
                    gradients, epoch, client_rank,
                    original_data=original_data, original_labels=original_labels
                )
                results[attack_type] = result
            elif attack_type == 'deep_leakage':
                result = self.perform_deep_leakage(
                    gradients, epoch, client_rank,
                    original_data=original_data, original_labels=original_labels
                )
                results[attack_type] = result
        
        return results