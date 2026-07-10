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
    
    def perform_gradient_inversion(self, gradients: Dict, epoch: int, client_rank: int = None,
                                  batch_size: int = 1, num_iterations: int = 1000,
                                  lr: float = 0.1, original_data: torch.Tensor = None,
                                  original_labels: torch.Tensor = None) -> Dict:
        """执行梯度反转攻击
        
        从梯度重建训练数据和标签
        
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
            'details': {}
        }
        
        try:
            if self.model is None:
                raise ValueError("模型未提供，无法进行梯度反转攻击")
            
            # 推断输入形状
            # LeNet5期望输入32x32（CIFAR）才能得到400维特征
            # 计算：32->28->14->10->5, 5*5*16=400
            input_shape = None
            for name, grad in gradients.items():
                if 'conv1' in name.lower() or ('conv' in name.lower() and 'weight' in name.lower()):
                    if len(grad.shape) >= 4:  # Conv2d weight: [out_channels, in_channels, H, W]
                        # 从梯度形状推断输入通道数
                        in_channels = grad.shape[1]
                        # 根据数据集类型确定输入大小
                        # CIFAR: 32x32, MNIST: 28x28
                        # 但LeNet5的fc层期望400维，所以必须用32x32
                        if self.data_loader:
                            data_name = str(self.data_loader).lower()
                            if 'cifar' in data_name:
                                input_shape = (batch_size, in_channels, 32, 32)
                            elif 'mnist' in data_name:
                                # MNIST用28x28，但需要调整模型或使用32x32
                                input_shape = (batch_size, in_channels, 32, 32)  # 统一使用32x32
                            else:
                                input_shape = (batch_size, in_channels, 32, 32)
                        else:
                            # 默认使用32x32（CIFAR标准）
                            input_shape = (batch_size, in_channels, 32, 32)
                        break
            
            if input_shape is None:
                # 默认使用CIFAR形状（32x32）
                input_shape = (batch_size, 3, 32, 32)
            
            # 初始化随机数据和标签
            dummy_data = torch.randn(input_shape, device=self.device, requires_grad=True)
            dummy_labels = torch.randint(0, self.n_class, (batch_size,), device=self.device)
            
            # 优化器
            optimizer = optim.Adam([dummy_data], lr=lr)
            
            # 将模型设置为训练模式（需要梯度）
            self.model.train()
            
            # 检查梯度键名是否匹配模型参数名
            model_param_names = set(name for name, _ in self.model.named_parameters())
            grad_names = set(gradients.keys())
            matching_names = model_param_names & grad_names
            
            if len(matching_names) == 0:
                # 尝试匹配：可能梯度键名格式不同（如 'conv1.0.weight' vs 'conv1.weight'）
                # 或者梯度是加密压缩后的，需要先处理
                if self.logger:
                    self.logger.warning(f'梯度键名与模型参数名不匹配。模型参数: {list(model_param_names)[:3]}..., 梯度键: {list(grad_names)[:3]}...')
                # 尝试直接使用梯度，假设形状匹配
                matching_names = grad_names
            
            best_loss = float('inf')
            best_data = None
            best_labels = None
            
            # 优化循环
            for iteration in range(num_iterations):
                optimizer.zero_grad()
                
                # 清除模型参数的梯度
                for param in self.model.parameters():
                    param.grad = None
                
                # 前向传播
                try:
                    output = self.model(dummy_data)
                    loss = nn.CrossEntropyLoss()(output, dummy_labels)
                except RuntimeError as e:
                    error_msg = str(e)
                    if 'mat1 and mat2 shapes cannot be multiplied' in error_msg:
                        # 维度不匹配，可能是输入大小不对
                        if iteration == 0 and self.logger:
                            # 尝试诊断问题
                            try:
                                with torch.no_grad():
                                    test_conv1 = self.model.conv1(dummy_data)
                                    test_conv2 = self.model.conv2(test_conv1)
                                    test_flat = test_conv2.view(test_conv2.size(0), -1)
                                    self.logger.warning(f'维度不匹配诊断: conv1输出={test_conv1.shape}, conv2输出={test_conv2.shape}, flatten={test_flat.shape}')
                                    self.logger.warning(f'模型fc.0期望输入: {self.model.fc[0].in_features}')
                            except:
                                pass
                        # 跳过此次迭代，不更新
                        continue
                    else:
                        if self.logger and iteration == 0:
                            self.logger.warning(f'前向传播失败: {e}，跳过此次迭代')
                        continue
                
                # 反向传播
                loss.backward()
                
                # 计算重建梯度与真实梯度的差异
                grad_diff = 0.0
                count = 0
                for name in matching_names:
                    try:
                        param = dict(self.model.named_parameters())[name]
                        if param.grad is not None and name in gradients:
                            true_grad = gradients[name]
                            # 确保在同一设备上
                            if isinstance(true_grad, torch.Tensor):
                                true_grad = true_grad.to(self.device)
                                # 确保形状匹配
                                if param.grad.shape == true_grad.shape:
                                    grad_diff += torch.nn.functional.mse_loss(param.grad, true_grad)
                                    count += 1
                    except KeyError:
                        # 参数名不在模型中，跳过
                        continue
                    except Exception as e:
                        if self.logger and iteration == 0:
                            self.logger.debug(f'处理梯度 {name} 时出错: {e}')
                        continue
                
                if count == 0:
                    # 如果没有匹配的梯度，只使用分类损失
                    total_loss = loss
                else:
                    # 总损失：分类损失 + 梯度差异
                    grad_diff = grad_diff / count if count > 0 else 0.0
                    total_loss = loss + grad_diff * 100.0
                
                # 更新
                optimizer.step()
                
                # 限制数据范围（图像应该在合理范围内）
                with torch.no_grad():
                    dummy_data.clamp_(min=-3.0, max=3.0)
                
                # 记录最佳结果
                if total_loss.item() < best_loss:
                    best_loss = total_loss.item()
                    best_data = dummy_data.detach().clone()
                    best_labels = dummy_labels.clone()
                
                # 检查是否需要在当前迭代步数保存中间结果
                current_iteration = iteration + 1
                should_save_iteration = (self.save_iteration_steps is not None) and (current_iteration in self.save_iteration_steps)
                if should_save_iteration:
                    # 保存当前迭代的重建结果
                    current_data = dummy_data.detach().clone()
                    current_labels = dummy_labels.clone()
                    try:
                        # 使用原始数据（如果有）或当前重建数据作为原始数据
                        vis_original_data = original_data if original_data is not None else current_data
                        vis_original_labels = original_labels if original_labels is not None else current_labels
                        self._visualize_reconstruction(
                            vis_original_data, current_data, vis_original_labels, current_labels,
                            epoch, client_rank, 'gradient_inversion', iteration_step=current_iteration
                        )
                    except Exception as e:
                        if self.logger:
                            self.logger.warning(f'保存迭代{current_iteration}的重建结果失败: {e}')
                
                # 每100次迭代输出一次
                if (iteration + 1) % 100 == 0 and self.logger:
                    self.logger.debug(f'  迭代 {iteration+1}/{num_iterations}, Loss: {total_loss.item():.6f}')
            
            # 检查是否有有效的结果
            if best_data is None or best_labels is None:
                # 所有迭代都失败了，使用初始值
                if self.logger:
                    self.logger.warning('所有迭代都失败，使用初始随机值作为重建结果')
                best_data = dummy_data.detach().clone()
                best_labels = dummy_labels.clone()
            
            # 重建的数据和标签
            reconstructed_data = best_data
            reconstructed_labels = best_labels
            
            # 计算标签准确率（如果有原始标签）
            label_accuracy = 0.0
            if original_labels is not None and reconstructed_labels is not None:
                try:
                    label_accuracy = (reconstructed_labels.cpu() == original_labels.cpu()).float().mean().item()
                except:
                    label_accuracy = 0.0
            
            attack_result['success'] = True
            attack_result['label_accuracy'] = label_accuracy
            attack_result['details'] = {
                'reconstructed_data_shape': list(reconstructed_data.shape) if reconstructed_data is not None else None,
                'reconstructed_labels': reconstructed_labels.cpu().tolist() if reconstructed_labels is not None else None,
                'best_loss': float(best_loss) if best_loss != float('inf') else None,
                'num_iterations': num_iterations,
                'all_iterations_failed': (best_data is None or best_labels is None)
            }
            
            # 保存重建结果
            if reconstructed_data is not None and reconstructed_labels is not None:
                # 根据攻击者类型和client_rank生成文件名
                if client_rank is not None:
                    identifier = f'client{client_rank}'
                else:
                    identifier = self.attacker_id if self.attacker_id else 'unknown'
                save_path = f'./cache/attack_results/gradient_inversion_epoch{epoch}_{identifier}.pkl'
                try:
                    torch.save({
                        'reconstructed_data': reconstructed_data.cpu(),
                        'reconstructed_labels': reconstructed_labels.cpu(),
                        'original_data': original_data.cpu() if original_data is not None else None,
                        'original_labels': original_labels.cpu() if original_labels is not None else None,
                        'epoch': epoch,
                        'client_rank': client_rank,
                        'attacker_id': self.attacker_id
                    }, save_path)
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f'保存重建结果失败: {e}')
                
                # 可视化：有原始数据用原始数据，对应轮次缺原始数据则用重建结果自身可视化
                should_save = (self.save_visualization_epochs is None) or (epoch in self.save_visualization_epochs)
                if should_save:
                    try:
                        if original_data is not None:
                            self._visualize_reconstruction(
                                original_data, reconstructed_data, original_labels, reconstructed_labels,
                                epoch, client_rank, 'gradient_inversion', iteration_step=None
                            )
                        else:
                            # 无原始数据，使用重建结果自身作为对比
                            self._visualize_reconstruction(
                                reconstructed_data, reconstructed_data, reconstructed_labels, reconstructed_labels,
                                epoch, client_rank, 'gradient_inversion', iteration_step=None
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
        """执行梯度深度泄露攻击（Deep Leakage from Gradients）
        
        使用更优化的方法从梯度重建训练数据
        
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
            'details': {}
        }
        
        try:
            if self.model is None:
                raise ValueError("模型未提供，无法进行深度泄露攻击")
            
            # 推断输入形状
            # LeNet5期望输入32x32（CIFAR）才能得到400维特征
            input_shape = None
            for name, grad in gradients.items():
                if 'conv1' in name.lower() or ('conv' in name.lower() and 'weight' in name.lower()):
                    if len(grad.shape) >= 4:  # Conv2d weight: [out_channels, in_channels, H, W]
                        in_channels = grad.shape[1]
                        # 统一使用32x32以确保维度匹配（LeNet5的fc层期望400维输入）
                        if self.data_loader:
                            data_name = str(self.data_loader).lower()
                            if 'cifar' in data_name or 'mnist' in data_name:
                                input_shape = (batch_size, in_channels, 32, 32)
                            else:
                                input_shape = (batch_size, in_channels, 32, 32)
                        else:
                            input_shape = (batch_size, in_channels, 32, 32)
                        break
            
            if input_shape is None:
                input_shape = (batch_size, 3, 32, 32)
            
            # 初始化：使用更好的初始化策略
            dummy_data = torch.randn(input_shape, device=self.device, requires_grad=True)
            # 使用标签的one-hot编码进行优化
            dummy_labels_onehot = torch.randn(batch_size, self.n_class, device=self.device, requires_grad=True)
            
            # 优化器（使用不同的学习率）
            optimizer = optim.Adam([
                {'params': [dummy_data], 'lr': lr},
                {'params': [dummy_labels_onehot], 'lr': lr * 10}
            ])
            
            self.model.train()
            
            # 检查梯度键名是否匹配模型参数名
            model_param_names = set(name for name, _ in self.model.named_parameters())
            grad_names = set(gradients.keys())
            matching_names = model_param_names & grad_names
            
            if len(matching_names) == 0:
                if self.logger:
                    self.logger.warning(f'梯度键名与模型参数名不匹配。模型参数: {list(model_param_names)[:3]}..., 梯度键: {list(grad_names)[:3]}...')
                matching_names = grad_names
            
            best_loss = float('inf')
            best_data = None
            best_labels = None
            
            # 优化循环
            for iteration in range(num_iterations):
                optimizer.zero_grad()
                
                # 清除模型参数的梯度
                for param in self.model.parameters():
                    param.grad = None
                
                # 将one-hot转换为标签
                dummy_labels = torch.softmax(dummy_labels_onehot, dim=1).argmax(dim=1)
                
                # 前向传播
                try:
                    output = self.model(dummy_data)
                    loss = nn.CrossEntropyLoss()(output, dummy_labels)
                except RuntimeError as e:
                    error_msg = str(e)
                    if 'mat1 and mat2 shapes cannot be multiplied' in error_msg:
                        # 维度不匹配，可能是输入大小不对
                        if iteration == 0 and self.logger:
                            # 尝试诊断问题
                            try:
                                with torch.no_grad():
                                    test_conv1 = self.model.conv1(dummy_data)
                                    test_conv2 = self.model.conv2(test_conv1)
                                    test_flat = test_conv2.view(test_conv2.size(0), -1)
                                    self.logger.warning(f'维度不匹配诊断: conv1输出={test_conv1.shape}, conv2输出={test_conv2.shape}, flatten={test_flat.shape}')
                                    self.logger.warning(f'模型fc.0期望输入: {self.model.fc[0].in_features}')
                            except:
                                pass
                        # 跳过此次迭代，不更新
                        continue
                    else:
                        if self.logger and iteration == 0:
                            self.logger.warning(f'前向传播失败: {e}，跳过此次迭代')
                        continue
                
                # 反向传播（只对loss进行反向传播以获取模型梯度）
                loss.backward(retain_graph=True)
                
                # 计算梯度匹配损失
                grad_diff = 0.0
                count = 0
                for name in matching_names:
                    try:
                        param = dict(self.model.named_parameters())[name]
                        if param.grad is not None and name in gradients:
                            true_grad = gradients[name]
                            if isinstance(true_grad, torch.Tensor):
                                true_grad = true_grad.to(self.device)
                                if param.grad.shape == true_grad.shape:
                                    grad_diff += torch.nn.functional.mse_loss(param.grad, true_grad)
                                    count += 1
                    except KeyError:
                        # 参数名不在模型中，跳过
                        continue
                    except Exception as e:
                        if self.logger and iteration == 0:
                            self.logger.debug(f'处理梯度 {name} 时出错: {e}')
                        continue
                
                if count > 0:
                    grad_diff = grad_diff / count
                else:
                    grad_diff = 0.0
                
                # 添加正则化
                tv_loss = self._total_variation_loss(dummy_data)
                
                # 总损失：梯度匹配损失（主要）+ 分类损失（辅助）+ 正则化
                # 由于loss已经反向传播过（使用retain_graph=True），我们需要清除梯度后重新计算
                # 这样可以避免对同一计算图进行两次反向传播
                
                # 清除所有梯度
                optimizer.zero_grad()
                
                # 重新前向传播计算loss（用于构建total_loss）
                output_new = self.model(dummy_data)
                loss_new = nn.CrossEntropyLoss()(output_new, dummy_labels)
                
                # 构建total_loss（grad_diff是标量，不依赖计算图，可以直接使用）
                total_loss = grad_diff * 1000.0 + loss_new * 0.1 + tv_loss * 0.01
                
                # 对total_loss进行反向传播（这是唯一一次反向传播）
                total_loss.backward()
                optimizer.step()
                
                # 限制数据范围
                with torch.no_grad():
                    dummy_data.clamp_(min=-3.0, max=3.0)
                
                # 记录最佳结果
                if total_loss.item() < best_loss:
                    best_loss = total_loss.item()
                    best_data = dummy_data.detach().clone()
                    best_labels = torch.softmax(dummy_labels_onehot.detach(), dim=1).argmax(dim=1).clone()
                
                # 检查是否需要在当前迭代步数保存中间结果
                current_iteration = iteration + 1
                should_save_iteration = (self.save_iteration_steps is not None) and (current_iteration in self.save_iteration_steps)
                if should_save_iteration:
                    # 保存当前迭代的重建结果
                    current_data = dummy_data.detach().clone()
                    current_labels = torch.softmax(dummy_labels_onehot.detach(), dim=1).argmax(dim=1).clone()
                    try:
                        # 使用原始数据（如果有）或当前重建数据作为原始数据
                        vis_original_data = original_data if original_data is not None else current_data
                        vis_original_labels = original_labels if original_labels is not None else current_labels
                        self._visualize_reconstruction(
                            vis_original_data, current_data, vis_original_labels, current_labels,
                            epoch, client_rank, 'deep_leakage', iteration_step=current_iteration
                        )
                    except Exception as e:
                        if self.logger:
                            self.logger.warning(f'保存迭代{current_iteration}的重建结果失败: {e}')
                
                if (iteration + 1) % 200 == 0 and self.logger:
                    self.logger.debug(f'  迭代 {iteration+1}/{num_iterations}, Loss: {total_loss.item():.6f}, '
                                    f'Grad Diff: {grad_diff.item():.6f}')
            
            # 检查是否有有效的结果
            if best_data is None or best_labels is None:
                # 所有迭代都失败了，使用初始值
                if self.logger:
                    self.logger.warning('所有迭代都失败，使用初始随机值作为重建结果')
                best_data = dummy_data.detach().clone()
                best_labels = torch.softmax(dummy_labels_onehot.detach(), dim=1).argmax(dim=1).clone()
            
            reconstructed_data = best_data
            reconstructed_labels = best_labels
            
            # 计算标签准确率
            label_accuracy = 0.0
            if original_labels is not None and reconstructed_labels is not None:
                try:
                    label_accuracy = (reconstructed_labels.cpu() == original_labels.cpu()).float().mean().item()
                except:
                    label_accuracy = 0.0
            
            attack_result['success'] = True
            attack_result['label_accuracy'] = label_accuracy
            attack_result['details'] = {
                'reconstructed_data_shape': list(reconstructed_data.shape) if reconstructed_data is not None else None,
                'reconstructed_labels': reconstructed_labels.cpu().tolist() if reconstructed_labels is not None else None,
                'best_loss': float(best_loss) if best_loss != float('inf') else None,
                'num_iterations': num_iterations,
                'all_iterations_failed': (best_data is None or best_labels is None)
            }
            
            # 保存重建结果
            if reconstructed_data is not None and reconstructed_labels is not None:
                # 根据攻击者类型和client_rank生成文件名
                if client_rank is not None:
                    identifier = f'client{client_rank}'
                else:
                    identifier = self.attacker_id if self.attacker_id else 'unknown'
                save_path = f'./cache/attack_results/deep_leakage_epoch{epoch}_{identifier}.pkl'
                try:
                    torch.save({
                        'reconstructed_data': reconstructed_data.cpu(),
                        'reconstructed_labels': reconstructed_labels.cpu(),
                        'original_data': original_data.cpu() if original_data is not None else None,
                        'original_labels': original_labels.cpu() if original_labels is not None else None,
                        'epoch': epoch,
                        'client_rank': client_rank,
                        'attacker_id': self.attacker_id
                    }, save_path)
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f'保存重建结果失败: {e}')
                
                # 可视化：有原始数据用原始数据，对应轮次缺原始数据则用重建结果自身可视化
                should_save = (self.save_visualization_epochs is None) or (epoch in self.save_visualization_epochs)
                if should_save:
                    try:
                        if original_data is not None:
                            self._visualize_reconstruction(
                                original_data, reconstructed_data, original_labels, reconstructed_labels,
                                epoch, client_rank, 'deep_leakage', iteration_step=None
                            )
                        else:
                            # 无原始数据，使用重建结果自身作为对比
                            self._visualize_reconstruction(
                                reconstructed_data, reconstructed_data, reconstructed_labels, reconstructed_labels,
                                epoch, client_rank, 'deep_leakage', iteration_step=None
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
                                 epoch: int, client_rank: int, attack_type: str, iteration_step: int = None):
        """可视化重建结果
        
        Args:
            original_data: 原始数据
            reconstructed_data: 重建数据
            original_labels: 原始标签
            reconstructed_labels: 重建标签
            epoch: 训练轮次
            client_rank: 客户端rank
            attack_type: 攻击类型
            iteration_step: 优化迭代步数（如果提供，会在文件名中包含此信息）
        """
        try:
            # 转换为numpy并反归一化（假设使用CIFAR归一化）
            def denormalize(tensor):
                mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
                std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1)
                return tensor * std + mean
            
            original_vis = denormalize(original_data.cpu())
            reconstructed_vis = denormalize(reconstructed_data.cpu())
            
            # 限制到[0,1]
            original_vis = torch.clamp(original_vis, 0, 1)
            reconstructed_vis = torch.clamp(reconstructed_vis, 0, 1)
            
            # 转换为numpy
            original_vis = original_vis.permute(0, 2, 3, 1).numpy()
            reconstructed_vis = reconstructed_vis.permute(0, 2, 3, 1).numpy()
            
            # 根据攻击者类型和client_rank生成标识符
            if client_rank is not None:
                # 外部攻击者或中间服务器攻击特定客户端时，使用client标识
                identifier = f'client{client_rank}'
            else:
                # 内部攻击者（中心服务器或中间服务器聚合梯度）使用attacker_id
                identifier = self.attacker_id if self.attacker_id else 'unknown'
            
            # 检查是否已经保存过原始图片（每个客户端/服务器在每个epoch的每次攻击只保存一次）
            original_image_key = (epoch, identifier, attack_type)
            should_save_original = original_image_key not in self.saved_original_images
            
            # 保存原始图片（单独保存，只在第一次保存）
            num_samples = min(original_data.size(0), 8)  # 最多显示8个样本
            if num_samples > 0 and should_save_original:
                # 保存原始图片
                fig_original, axes_original = plt.subplots(1, num_samples, figsize=(2*num_samples, 2))
                if num_samples == 1:
                    axes_original = [axes_original]
                for i in range(num_samples):
                    axes_original[i].imshow(original_vis[i])
                    axes_original[i].set_title(f'原始\n标签: {original_labels[i].item()}')
                    axes_original[i].axis('off')
                plt.tight_layout()
                # 原始图片文件名不包含迭代步数，因为每个攻击只保存一次
                original_save_path = f'./cache/attack_results/{attack_type}_epoch{epoch}_{identifier}_original.png'
                plt.savefig(original_save_path, dpi=150, bbox_inches='tight')
                plt.close(fig_original)
                # 标记为已保存
                self.saved_original_images.add(original_image_key)
                if self.logger:
                    self.logger.info(f'  原始图片已保存: {original_save_path}')
            
            # 创建对比可视化（原始 vs 重建）
            fig, axes = plt.subplots(2, num_samples, figsize=(2*num_samples, 4))
            
            if num_samples == 1:
                axes = axes.reshape(2, 1)
            
            for i in range(num_samples):
                # 原始图像
                axes[0, i].imshow(original_vis[i])
                axes[0, i].set_title(f'原始\n标签: {original_labels[i].item()}')
                axes[0, i].axis('off')
                
                # 重建图像
                axes[1, i].imshow(reconstructed_vis[i])
                pred_label = reconstructed_labels[i].item()
                true_label = original_labels[i].item()
                match = "✓" if pred_label == true_label else "✗"
                axes[1, i].set_title(f'重建 {match}\n标签: {pred_label}')
                axes[1, i].axis('off')
            
            plt.tight_layout()
            iter_suffix = f'_iter{iteration_step}' if iteration_step is not None else ''
            # 根据攻击者类型和client_rank生成文件名
            if client_rank is not None:
                # 外部攻击者或中间服务器攻击特定客户端时，使用client标识
                identifier = f'client{client_rank}'
            else:
                # 内部攻击者（中心服务器或中间服务器聚合梯度）使用attacker_id
                identifier = self.attacker_id if self.attacker_id else 'unknown'
            save_path = f'./cache/attack_results/{attack_type}_epoch{epoch}_{identifier}{iter_suffix}_visualization.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            if self.logger:
                self.logger.info(f'  可视化结果已保存: {save_path}')
                
        except Exception as e:
            if self.logger:
                self.logger.warning(f'可视化失败: {e}')


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
