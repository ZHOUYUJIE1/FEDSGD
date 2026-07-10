
import math
from numpy import dtype
import torch
import torch.nn.functional as F
import hashlib
import struct

from base_compressor import Compressor


class TopkCompressor(Compressor):
    """ Compressor for federated communication

        Top-k gradient or weights selection with adaptive sparsity based on cosine similarity
        and encrypted compression using shared key between clients and center server

        Args:
            compress_ratio (float): base compress ratio
            min_sparsity_ratio (float): minimum sparsity ratio (default: 0.1)
            max_sparsity_ratio (float): maximum sparsity ratio (default: 0.9)
            encryption_key (str, optional): shared encryption key for compression (only known to clients and center server)
    """
    def __init__(self, compress_ratio, min_sparsity_ratio=0.1, max_sparsity_ratio=0.9, bate=0.3, encryption_key=None,unlearn_mode='unlearn',logger=None, compression_mode='adaptive'):
        """
        Args:
            compress_ratio (float): base compress ratio (用于fixed模式)
            min_sparsity_ratio (float): minimum sparsity ratio (default: 0.1)
            max_sparsity_ratio (float): maximum sparsity ratio (default: 0.9)
            bate (float): 遗忘学习时的衰减系数
            encryption_key (str, optional): shared encryption key for compression
            unlearn_mode (str): 遗忘学习模式，'unlearn'/'invert'/'normal'
            logger: 日志记录器
            compression_mode (str): 压缩模式
                - 'adaptive': 动态稀疏化压缩（根据global_gradient计算adaptive_sparsity）
                - 'fixed': 固定稀疏度压缩（使用compress_ratio）
                - 'none': 不使用稀疏压缩（返回完整梯度）
        """
        self.compress_ratio = compress_ratio if compress_ratio <= 1.0 else 1.0 / compress_ratio
        self.min_sparsity_ratio = min_sparsity_ratio
        self.max_sparsity_ratio = max_sparsity_ratio
        self.index_dtype = torch.int64
        self.value_dtype = torch.float32
        self.bate = bate
        self.encryption_key = encryption_key  # 压缩密钥，只有客户端和中心服务器知道
        self.unlearn_mode=unlearn_mode # 遗忘学习模式，默认是unlearn，也可以是normal,也可以是invert作为对照试验
        self.logger=logger # 日志记录器
        self.compression_mode = compression_mode  # 压缩模式：'adaptive', 'fixed', 'none'
        
        # 验证compression_mode参数
        if compression_mode not in ['adaptive', 'fixed', 'none']:
            raise ValueError(f"compression_mode必须是'adaptive', 'fixed'或'none'之一，当前值: {compression_mode}")
    def _generate_layer_key(self, layer_name, tensor_shape):

        """根据层名称和形状生成该层的加密密钥
        
        Args:
            layer_name: 参数层名称
            tensor_shape: 张量形状
            
        Returns:
            torch.Tensor: 用于加密的随机掩码
        """
        if self.encryption_key is None:
            return None
        
        # 使用密钥和层信息生成确定性的随机掩码
        key_string = f"{self.encryption_key}_{layer_name}_{str(tensor_shape)}"
        key_hash = hashlib.sha256(key_string.encode()).digest()
        
        # 使用哈希值作为随机种子生成掩码
        seed = struct.unpack('Q', key_hash[:8])[0]  # 取前8字节作为种子
        generator = torch.Generator()
        generator.manual_seed(seed)
        
        # 生成与张量形状相同的随机掩码（范围在[-1, 1]）
        mask = torch.rand(tensor_shape, generator=generator) * 2.0 - 1.0
        return mask
    
    def _encrypt_compress_layer(self, tensor, layer_name):

        """对单个层的梯度进行加密压缩
        
        Args:
            tensor: 稀疏化后的梯度张量
            layer_name: 层名称
            
        Returns:
            encrypted_data: 加密压缩后的数据（可以是字节流或加密张量）
        """
        if self.encryption_key is None:
            # 如果没有密钥，直接返回原始张量（向后兼容）
            return tensor
        
        # 生成该层的加密掩码
        mask = self._generate_layer_key(layer_name, tensor.shape)
        mask = mask.to(tensor.device)
        
        # 加密：添加掩码扰动（可逆操作，但中间服务器不知道掩码无法解密）
        encrypted = tensor + mask * 0.1  # 0.1是扰动强度，可作为防御性噪声
        
        # 量化压缩（进一步压缩数据，同时引入可控损耗）
        # 将浮点数量化到较低精度，减少传输量
        scale = 1000.0  # 量化缩放因子
        quantized = torch.round(encrypted * scale) / scale
        
        return quantized
    
    def _decrypt_decompress_layer(self, encrypted_data, layer_name, original_shape):

        """对加密压缩的层进行解密解压缩
        
        Args:
            encrypted_data: 加密压缩后的数据
            layer_name: 层名称
            original_shape: 原始张量形状
            
        Returns:
            decrypted_tensor: 解密解压缩后的张量（可能有损耗）
        """
        if self.encryption_key is None:
            # 如果没有密钥，直接返回（向后兼容）
            return encrypted_data
        
        # 生成该层的解密掩码（与加密时相同）
        mask = self._generate_layer_key(layer_name, original_shape)
        mask = mask.to(encrypted_data.device)
        
        # 解密：移除掩码扰动
        decrypted = encrypted_data - mask * 0.1
        
        # 注意：量化造成的损耗无法完全恢复，这可以作为防御性扰动
        
        return decrypted

    def _compute_adaptive_sparsity(self, local_grad, global_grad,unlearning_flag=False):

        """Compute adaptive sparsity ratio based on cosine similarity
        
        Args:
            local_grad (torch.Tensor): local gradient tensor
            global_grad (torch.Tensor): global gradient tensor (normalized direction)
        
        Returns:
            float: adaptive sparsity ratio
        """
        # Flatten tensors for computation
        local_flat = local_grad.view(-1)
        global_flat = global_grad.detach().clone().view(-1)
        
        # Normalize local gradient
        local_norm = torch.norm(local_flat)
        if local_norm > 0:
            local_normalized = local_flat / local_norm
        else:
            local_normalized = local_flat

        global_normalized = global_flat
        
        # Compute cosine similarity
        cosine_sim = torch.dot(local_normalized, global_normalized)
        
        # Apply sigmoid to cosine similarity
        sigmoid_sim = torch.sigmoid(cosine_sim)
        
        # Map sigmoid result to sparsity ratio range
        # sigmoid_sim=sigmoid_sim.clamp(0.0,0.1)

        if unlearning_flag:
            if self.unlearn_mode == 'unlearn':
                # self.logger.info(f"Unlearning mode: unlearning")
                base_sparsity = self.min_sparsity_ratio + (1-sigmoid_sim) * (self.max_sparsity_ratio - self.min_sparsity_ratio)
                adaptive_sparsity = max(self.min_sparsity_ratio,self.bate*base_sparsity)
            elif self.unlearn_mode == 'invert':
                # self.logger.info(f"Unlearning mode: invert")
                adaptive_sparsity = self.min_sparsity_ratio + sigmoid_sim * (self.max_sparsity_ratio - self.min_sparsity_ratio)
            else:
                raise ValueError(f"Invalid unlearn mode: {self.unlearn_mode}")
        else:
            # self.logger.info(f"Unlearning mode: normal")
            # adaptive_sparsity = self.min_sparsity_ratio + (1-sigmoid_sim) * (self.max_sparsity_ratio - self.min_sparsity_ratio)
            adaptive_sparsity = self.min_sparsity_ratio + (sigmoid_sim) * (self.max_sparsity_ratio - self.min_sparsity_ratio)
            # adaptive_sparsity = min(max(0.5*sigmoid_sim+0.2,self.min_sparsity_ratio),self.max_sparsity_ratio)
        return adaptive_sparsity.item() if hasattr(adaptive_sparsity, "item") else float(adaptive_sparsity)

    def compress_tensor(self, tensor, global_gradient=None, unlearning_flag=False):

        """compress tensor into (values, indices)
        
        Note: Only the local gradient (tensor) is compressed. The global_gradient
        is used ONLY for computing adaptive sparsity ratio and is NOT compressed.

        Args:
            tensor (torch.Tensor): local gradient tensor (WILL BE COMPRESSED)
            global_gradient (torch.Tensor, optional): global gradient normalized direction.
                                                      Used ONLY for sparsity calculation, NOT compressed.
                                                      If None, use base compress_ratio.
            unlearning_flag (bool): 遗忘标志，True表示遗忘学习

        Returns:
            tuple: (values, indices) - compressed values and indices of LOCAL gradient only
                   If compression_mode='none', returns all values and indices (no compression)
        """
        if torch.is_tensor(tensor):
            tensor = tensor.detach()
        else:
            raise TypeError(
                "Invalid type error, expecting {}, but get {}".format(
                    torch.Tensor, type(tensor)))

        # 保存原始形状
        original_shape = tensor.shape
        numel = tensor.numel()
        tensor_flat = tensor.view(-1)
        
        # 根据compression_mode选择不同的压缩方式
        if self.compression_mode == 'none':
            # 不进行稀疏压缩，返回完整梯度
            indices = torch.arange(numel, dtype=self.index_dtype, device=tensor_flat.device)
            values = tensor_flat.to(dtype=self.value_dtype)
            return values, indices
        
        elif self.compression_mode == 'fixed':
            # 固定稀疏度压缩：使用compress_ratio
            top_k_samples = int(math.ceil(numel * (1 - self.compress_ratio)))
            importance = tensor_flat.abs()
            _, indices = torch.topk(importance,
                                    top_k_samples,
                                    0,
                                    largest=True,
                                    sorted=False)
            values = tensor_flat[indices]
            values = values.to(dtype=self.value_dtype)
            indices = indices.to(dtype=self.index_dtype)
            return values, indices
        
        elif self.compression_mode == 'adaptive':
            # 动态稀疏化压缩：根据global_gradient计算adaptive_sparsity
            if global_gradient is not None:
                if not torch.is_tensor(global_gradient):
                    global_gradient = torch.tensor(global_gradient, dtype=tensor.dtype, device=tensor.device)
                # Ensure global_gradient has the same shape as original tensor
                if global_gradient.shape != original_shape:
                    global_gradient = global_gradient.view(original_shape)
                # 计算自适应稀疏度（使用原始形状）
                if unlearning_flag:
                    adaptive_sparsity = self._compute_adaptive_sparsity(tensor, global_gradient, unlearning_flag=True)
                else:
                    adaptive_sparsity = self._compute_adaptive_sparsity(tensor, global_gradient, unlearning_flag=False)
                top_k_samples = int(math.ceil(numel * (adaptive_sparsity)))
            else:
                # Fall back to base compress_ratio if no global gradient provided
                top_k_samples = int(math.ceil(numel * (self.compress_ratio)))
            
            importance = tensor_flat.abs()
            _, indices = torch.topk(importance,
                                    top_k_samples,
                                    0,
                                    largest=True,
                                    sorted=False)
            values = tensor_flat[indices]
            values = values.to(dtype=self.value_dtype)
            indices = indices.to(dtype=self.index_dtype)
            return values, indices
        elif self.compression_mode == 'reverse':
            # 动态稀疏化压缩：根据global_gradient计算adaptive_sparsity
            if global_gradient is not None:
                if not torch.is_tensor(global_gradient):
                    global_gradient = torch.tensor(global_gradient, dtype=tensor.dtype, device=tensor.device)
                # Ensure global_gradient has the same shape as original tensor
                if global_gradient.shape != original_shape:
                    global_gradient = global_gradient.view(original_shape)
                # 计算自适应稀疏度（使用原始形状）
                if unlearning_flag:
                    adaptive_sparsity = self._compute_adaptive_sparsity(tensor, global_gradient, unlearning_flag=True)
                else:
                    adaptive_sparsity = self._compute_adaptive_sparsity(tensor, global_gradient, unlearning_flag=False)
                top_k_samples = int(math.ceil(numel * (1-adaptive_sparsity)))
            
            else:
                # Fall back to base compress_ratio if no global gradient provided
                top_k_samples = int(math.ceil(numel * (1-self.compress_ratio)))
            
            importance = tensor_flat.abs()
            _, indices = torch.topk(importance,
                                    top_k_samples,
                                    0,
                                    largest=True,
                                    sorted=False)
            values = tensor_flat[indices]
            values = values.to(dtype=self.value_dtype)
            indices = indices.to(dtype=self.index_dtype)
            return values, indices
        else:
            raise ValueError(f"不支持的compression_mode: {self.compression_mode}")

    def decompress_tensor(self, values, indices, shape):

        """decompress tensor"""
        device = values.device
        de_tensor = torch.zeros(size=shape, dtype=self.value_dtype, device=device).view(-1)
        de_tensor = de_tensor.index_put_([indices], values,
                                         accumulate=True).view(shape)
        return de_tensor

    # def compress_with_encryption(self, named_grads):
    #     """对命名梯度进行加密压缩（按层压缩）
        
    #     Args:
    #         named_grads: 字典，{layer_name: gradient_tensor}
            
    #     Returns:
    #         encrypted_grads: 字典，{layer_name: encrypted_compressed_tensor}
    #     """
    #     if self.compression_mode == 'none' or self.encryption_key is None:
    #         return named_grads
    #     encrypted_grads = {}
    #     for layer_name, grad in named_grads.items():
    #         encrypted_grads[layer_name] = self._encrypt_compress_layer(grad, layer_name)
    #     return encrypted_grads

    #deepseek
    def compress_with_encryption(self, named_grads, force_encrypt=False):

        """对命名梯度进行加密压缩
        
        Args:
            named_grads: 字典，{layer_name: gradient_tensor}
            force_encrypt: 是否强制加密（即使compression_mode='none'）
            
        Returns:
            encrypted_grads: 字典，{layer_name: encrypted_compressed_tensor}
        """
        # 如果压缩模式为'none'且不强制加密，直接返回
        if self.compression_mode == 'none' and not force_encrypt:
            return named_grads
        
        # 如果没有密钥，直接返回
        if self.encryption_key is None:
            return named_grads
        
        # 进行加密
        encrypted_grads = {}
        for layer_name, grad in named_grads.items():
            encrypted_grads[layer_name] = self._encrypt_compress_layer(grad, layer_name)
        
        return encrypted_grads
    
    def decompress_with_decryption(self, encrypted_grads, original_shapes, encrypted=True):

        """对加密压缩的梯度进行解密解压缩（按层解压）
        
        Args:
            encrypted_grads: 字典，{layer_name: encrypted_compressed_tensor}
            original_shapes: 字典，{layer_name: original_shape}
            encrypted (bool): 指示是否已加密（compression_mode='none' 时为 False）
            
        Returns:
            decrypted_grads: 字典，{layer_name: decrypted_tensor}（可能有损耗）
        """
        # 如果未加密或无密钥，直接返回原梯度
        if (not encrypted) or (self.encryption_key is None):
            return encrypted_grads
        
        decrypted_grads = {}
        for layer_name, encrypted in encrypted_grads.items():
            original_shape = original_shapes[layer_name]
            decrypted_grads[layer_name] = self._decrypt_decompress_layer(
                encrypted, layer_name, original_shape
            )
        return decrypted_grads

    def compress(self, parameters, global_gradients=None):

        """compress model

        Args:
            parameters: list of parameter tensors or model parameters
            global_gradients (list, optional): list of global gradient tensors (normalized directions).
                                              If None, use base compress_ratio for all parameters.

        Returns:
            tuple: list(values) and list(indices).
        """
        values_list = []
        indices_list = []
        
        # Convert parameters to list if needed
        if not isinstance(parameters, (list, tuple)):
            parameters = list(parameters)
        
        for i, param in enumerate(parameters):
            # Get corresponding global gradient if provided
            global_grad = None
            if global_gradients is not None:
                if i < len(global_gradients):
                    global_grad = global_gradients[i]
            
            values, indices = self.compress_tensor(param, global_grad)
            values_list.append(values)
            indices_list.append(indices)

        return values_list, indices_list

    def decompress(self, shape_list, values_list, indices_list):

        """decompress model

        Args:
            shape_list (list[tuple]): The shape of every corresponding tensor.
            values_list (list[torch.Tensor]): list(values).
            indices_list (list[torch.Tensor]): list(indices).
        """
        parameters_layer_list = []
        for shape, values, indices in zip(shape_list, values_list,
                                          indices_list):
            de_tensor = self.decompress_tensor(values, indices, shape)
            parameters_layer_list.append(de_tensor.view(-1))

        parameters = torch.cat(parameters_layer_list)

        return parameters
