import os
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler


class FedLogger:
    """联邦学习日志记录器
    
    支持同时输出到控制台和文件，支持日志轮转
    """
    
    def __init__(self, log_dir='./log', log_name='fedsgd', level=logging.INFO):
        """
        Args:
            log_dir: 日志文件目录
            log_name: 日志文件名前缀
            level: 日志级别
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # 创建日志文件名（带时间戳）
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'{log_name}_{timestamp}.log')
        
        # 配置日志格式
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        date_format = '%Y-%m-%d %H:%M:%S'
        
        # 创建logger
        self.logger = logging.getLogger('FedSGD')
        self.logger.setLevel(level)
        
        # 避免重复添加handler
        if not self.logger.handlers:
            # 文件handler（带轮转，每个文件最大10MB，保留5个备份）
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5,
                encoding='utf-8'
            )
            file_handler.setLevel(level)
            file_formatter = logging.Formatter(log_format, date_format)
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
            
            # 控制台handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
            console_formatter = logging.Formatter(log_format, date_format)
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
        
        # 记录日志系统初始化
        self.logger.info('='*60)
        self.logger.info('FedSGD 日志系统初始化')
        self.logger.info(f'日志文件: {log_file}')
        self.logger.info('='*60)
    
    def info(self, message):
        """记录INFO级别日志"""
        self.logger.info(message)
    
    def debug(self, message):
        """记录DEBUG级别日志"""
        self.logger.debug(message)
    
    def warning(self, message):
        """记录WARNING级别日志"""
        self.logger.warning(message)
    
    def error(self, message):
        """记录ERROR级别日志"""
        self.logger.error(message)
    
    def critical(self, message):
        """记录CRITICAL级别日志"""
        self.logger.critical(message)
    
    def log_config(self, config_dict):
        """记录配置信息"""
        self.logger.info('实验配置:')
        for key, value in config_dict.items():
            self.logger.info(f'  {key}: {value}')
    
    def log_epoch_start(self, epoch, total_epochs):
        """记录epoch开始"""
        self.logger.info('='*60)
        self.logger.info(f'Epoch {epoch}/{total_epochs} 开始')
        self.logger.info('='*60)
    
    def log_epoch_end(self, epoch, accuracy, loss=None):
        """记录epoch结束"""
        self.logger.info('='*60)
        self.logger.info(f'Epoch {epoch} 完成')
        if loss is not None:
            self.logger.info(f'  平均损失: {loss:.6f}')
        self.logger.info(f'  测试准确率: {accuracy*100:.2f}%')
        self.logger.info('='*60)
    
    def log_client_train(self, client_id, loss, accuracy, sparsity, compressed=False):
        """记录客户端训练信息"""
        compression_status = "已加密压缩" if compressed else "未压缩"
        self.logger.info(
            f'[Client {client_id:>2}] Loss: {loss:>8.6f}, '
            f'Accuracy: {accuracy:>6.4f}, Sparsity: {sparsity:>6.2f}% ({compression_status})'
        )
    
    def log_server_aggregate(self, server_type='center', info=None):
        """记录服务器聚合信息"""
        if server_type == 'center':
            self.logger.info('[Center Server] 聚合所有中间服务器的梯度')
        elif server_type == 'intermediate':
            if info:
                self.logger.info(
                    f'[Server1 {info.get("rank", "?")}] '
                    f'聚合了 {info.get("active_clients", 0)} 个活跃客户端 '
                    f'({info.get("forgotten_clients", 0)} 个已遗忘), '
                    f'总样本数: {info.get("total_samples", 0)}'
                )
    
    def log_gradient_info(self, grad_norm):
        """记录梯度信息"""
        self.logger.info(f'[Gradient Info] Gradient L2 Norm: {grad_norm:.6f}')
    
    def log_unlearning(self, message):
        """记录遗忘学习信息"""
        self.logger.info(f'[Unlearning] {message}')
    
    def log_data_distribution(self, client_id, train_samples, test_samples, train_classes, test_classes):
        """记录数据分布信息"""
        train_nonzero = sum(1 for c in train_classes if c > 0)
        test_nonzero = sum(1 for c in test_classes if c > 0)
        self.logger.info(
            f'  Client {client_id}: 训练数据 {train_samples} 条 ({train_nonzero} 个类别), '
            f'测试数据 {test_samples} 条 ({test_nonzero} 个类别)'
        )
    
    def log_attack(self, attacker_type, attacker_id, epoch, attack_results, attack_stage=None, server1_rank=None):
        """记录攻击信息
        
        Args:
            attacker_type: 攻击者类型 ('external', 'intermediate_server', 'center_server')
            attacker_id: 攻击者ID
            epoch: 训练轮次
            attack_results: 攻击结果字典
            attack_stage: 攻击阶段（如 'encrypted', 'decrypted'，仅用于中心服务器）
            server1_rank: 中间服务器rank（如果适用）
        """
        prefix = f'[Attack] {attacker_type.upper()}'
        if attacker_id:
            prefix += f' ({attacker_id})'
        if server1_rank is not None:
            prefix += f' [Server1 {server1_rank}]'
        if attack_stage:
            prefix += f' [{attack_stage.upper()}]'
        
        self.logger.info(f'{prefix} Epoch {epoch} - 攻击结果:')
        
        for attack_type, result in attack_results.items():
            if isinstance(result, dict):
                success = result.get('success', False)
                
                # 对于梯度反转和深度泄露攻击，显示标签准确率
                if attack_type in ['gradient_inversion', 'deep_leakage']:
                    label_accuracy = result.get('label_accuracy', 0.0)
                    self.logger.info(
                        f'  {attack_type}: success={success}, 标签准确率={label_accuracy*100:.2f}%'
                    )
                    details = result.get('details', {})
                    if 'reconstructed_labels' in details:
                        self.logger.info(f'    重建标签: {details["reconstructed_labels"]}')
                    if 'best_loss' in details:
                        self.logger.info(f'    最佳损失: {details["best_loss"]:.6f}')
                else:
                    # 其他攻击类型
                    confidence = result.get('confidence', 0.0)
                    self.logger.info(
                        f'  {attack_type}: success={success}, confidence={confidence:.4f}'
                    )
                    details = result.get('details', {})
                    if 'inferred_attributes' in details:
                        attrs = details['inferred_attributes']
                        if 'likely_classes' in attrs:
                            self.logger.info(f'    推断的类别: {attrs["likely_classes"]}')
                        if 'likely_non_iid' in attrs:
                            self.logger.info(f'    推断为非IID: {attrs["likely_non_iid"]}')


# 全局日志实例（单例模式）
_global_logger = None


def get_logger(log_dir='./log', log_name='fedsgd', level=logging.INFO):
    """获取全局日志实例"""
    global _global_logger
    if _global_logger is None:
        _global_logger = FedLogger(log_dir=log_dir, log_name=log_name, level=level)
    return _global_logger


def set_logger(logger):
    """设置全局日志实例（用于测试或自定义）"""
    global _global_logger
    _global_logger = logger
