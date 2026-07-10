from abc import ABC

from torch import nn
import torch.nn.functional as F


class LeNet5(nn.Module, ABC):
    def __init__(self, in_dim, n_class):
        super(LeNet5, self).__init__()  # super用法:继承父类nn.Model的属性，并用父类的方法初始化这些属性

        self.conv1 = nn.Sequential(
            # nn.Conv2d(in_dim, 6, 5, 1, 2),  # out_dim=6, kernel_size=5, stride=1, padding=2
            nn.Conv2d(in_dim, 6, 5, 1, 0),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # kernel_size=2, padding=2
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5, 1, 0),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        # 使用自适应平均池化，将特征图统一到5x5，这样无论输入是28x28(MNIST)还是32x32(CIFAR)都能得到400维特征
        self.adaptive_pool = nn.AdaptiveAvgPool2d((5, 5))
        self.fc = nn.Sequential(
            nn.Linear(400, 120),  # in_features=400 (16*5*5), out_features=120
            nn.Linear(120, 84),
            nn.Linear(84, n_class)
        )

    def forward(self, x):
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        # 使用自适应池化统一特征图大小到5x5
        out_conv2 = self.adaptive_pool(out_conv2)
        out_conv = out_conv2.view(out_conv2.size(0), -1)

        out = self.fc(out_conv)
        return out


def lenet5(n_class=10, in_dim=3):
    """ return a LeNet 5 object
    
    Args:
        n_class: 类别数，默认10（CIFAR10/MNIST），CIFAR100需要设置为100
        in_dim: 输入通道数，默认3（CIFAR-10/CIFAR-100），MNIST需要设置为1
    """
    return LeNet5(in_dim, n_class)


class BasicBlock(nn.Module):
    """ResNet 基础残差块"""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet18(nn.Module, ABC):
    """ResNet18 模型，适用于 CIFAR-10/100 和 MNIST"""
    def __init__(self, in_dim, n_class):
        super(ResNet18, self).__init__()
        self.in_planes = 64

        # 第一个卷积层：根据输入通道数调整
        self.conv1 = nn.Conv2d(in_dim, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # ResNet 层
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        
        # 全局平均池化和分类层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, n_class)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def resnet18(n_class=10, in_dim=3):
    """返回一个 ResNet18 对象
    
    Args:
        n_class: 类别数，默认10（CIFAR10/MNIST），CIFAR100需要设置为100
        in_dim: 输入通道数，默认3（CIFAR-10/CIFAR-100），MNIST需要设置为1
    """
    return ResNet18(in_dim, n_class)



class CIFAR10CNN_V2(nn.Module):
    """CIFAR10CNN_V2 模型，适用于 CIFAR-10"""
    def __init__(self, in_dim, n_class):
        super(CIFAR10CNN_V2, self).__init__()
        
        # 第一卷积层：通道数 in_dim→32，卷积核 3×3
        self.conv1 = nn.Conv2d(in_dim, 32, kernel_size=3, padding=1)  # 保持32×32尺寸
        
        # 第二卷积层：通道数 32→64，卷积核 3×3
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # 最大池化层：2×2，步长2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 32×32 → 16×16
        
        # 三个全连接层（都使用ReLU激活）
        # 计算展平后的尺寸：经过池化后为16×16，通道64
        self.fc1 = nn.Linear(64 * 16 * 16, 512)  # 输入: 64*16*16=16384
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, n_class)
        
        # Dropout（可选，用于防止过拟合）
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        # 卷积层1 + ReLU
        x = F.relu(self.conv1(x))  # 输出: [batch, 32, 32, 32]
        
        # 卷积层2 + ReLU
        x = F.relu(self.conv2(x))  # 输出: [batch, 64, 32, 32]
        
        # 最大池化层
        x = self.pool(x)  # 输出: [batch, 64, 16, 16]
        
        # 展平
        x = x.view(-1, 64 * 16 * 16)
        
        # 全连接层1 + ReLU
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        # 全连接层2 + ReLU
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        # 全连接层3（输出层，不加ReLU）
        x = self.fc3(x)
        return x


def cifar10cnn_v2(n_class=10, in_dim=3):
    """返回一个 CIFAR10CNN_V2 对象
    
    Args:
        n_class: 类别数，默认10（CIFAR10/MNIST），CIFAR100需要设置为100
        in_dim: 输入通道数，默认3（CIFAR-10/CIFAR-100），MNIST需要设置为1
    """
    return CIFAR10CNN_V2(in_dim, n_class)

