#ResNet
#https://arxiv.org/pdf/1512.03385
#指导思想，残差学习
#这个用pytorch实现，前面已经手搓过相应函数，这次就不继续了


import torch
import torch.nn as nn
# 基于PyTorch实现ResNet-18模型

# 定义BasicBlock类，实现ResNet-18的残差结构
class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride, downsample):
        """
        参数说明：
        in_channel: 输入特征图的通道数
        out_channel: 输出特征图的通道数
        stride: 卷积步长（控制空间分辨率变化）
        downsample: 是否需要下采样（用于调整输入与输出维度差异）
        """
        # 调用父类nn.Module的构造函数
        super(BasicBlock, self).__init__()
        
        # 第一个卷积层（3x3卷积核）
        self.conv1 = nn.Conv2d(
            in_channels=in_channel, 
            out_channels=out_channel, 
            kernel_size=3, 
            stride=stride,  # 控制空间分辨率变化
            padding=1,      # 保持特征图尺寸不变（当stride=1时）
            bias=False      # 与unet不一样的是，卷积层不使用偏置，因为BN会处理偏移
        )
        
        # 批归一化层（加速训练并稳定模型）
        self.bn1 = nn.BatchNorm2d(out_channel)
        
        # 激活函数
        self.relu = nn.ReLU()
        
        # 第二个卷积层（3x3卷积核）
        self.conv2 = nn.Conv2d(
            in_channels=out_channel, 
            out_channels=out_channel, 
            kernel_size=3, 
            stride=1,     # 固定步长为1
            padding=1,    # 保持特征图尺寸
            bias=False
        )
        
        # 第二个批归一化层
        self.bn2 = nn.BatchNorm2d(out_channel)
        
        # 下采样模块（当输入与输出维度不匹配时使用）
        self.downsample = downsample

    def forward(self, x):
        # 如果下采样操作不为空，则需要对输入进行下采样得到捷径分支的输出
        if self.downsample is not None:
            residual = self.downsample(x)
        else:  # 保存输入数据，便于后面进行残差链接
            residual = x
        
        # 前向传播主路径
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 残差连接核心：将主路径输出与调整后的输入相加
        out += residual 
        out = self.relu(out)  # 残差相加后再relu
        
        return out

# 定义完整的ResNet-18网络结构
class ResNet_18(nn.Module):
    def __init__(self, num_classes):
        """
        参数：
        num_classes: 分类任务的类别数量
        """
        # 调用父类nn.Module的构造函数
        super(ResNet_18, self).__init__()
        
        # 第一个卷积层（输入通道数设为4可能需要根据实际输入调整）
        self.conv1 = nn.Conv2d(
            in_channels=3,      # 输入通道数为3（需根据实际输入维度修改）
            out_channels=64,    # 输出通道数
            kernel_size=7,      # 大卷积核用于特征提取
            stride=2,           # 初始空间分辨率减半
            padding=3,          # 保持尺寸计算：(7-1)/2=3
            bias=False
        )
        
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(  # 初始最大池化进一步减半空间尺寸
            kernel_size=3, 
            stride=2, 
            padding=1
        )
        
        # 创建四个残差层，分别对应resnet18的四个stage
        # 第一个残差层，输出通道数64，残差块个数2，不进行下采样
        self.layer1 = nn.Sequential(
            BasicBlock(64, 64, 1, None),
            BasicBlock(64, 64, 1, None)
        )
        # 第二个残差层，输出通道数128，残差块个数2，步长为2，进行下采样
        self.layer2 = nn.Sequential(
            BasicBlock(
                64, 128, 2,
                # 下采样模块：1x1卷积调整通道并下采样
                downsample=nn.Sequential(
                    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=2, bias=False),
                    nn.BatchNorm2d(128)
                )
            ),
            BasicBlock(128, 128, 1, None)
        )
        # 第三个残差层，输出通道数256，残差块个数2，步长为2，进行下采样
        self.layer3 = nn.Sequential(
            BasicBlock(
                128, 256, 2,
                downsample=nn.Sequential(
                    nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=2, bias=False),
                    nn.BatchNorm2d(256)
                )
            ),
            BasicBlock(256, 256, 1, None)
        )
        # 第四个残差层，输出通道数512，残差块个数2，步长为2，进行下采样
        self.layer4 = nn.Sequential(
            BasicBlock(
                256, 512, 2,
                downsample=nn.Sequential(
                    nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=2, bias=False),
                    nn.BatchNorm2d(512)
                )
            ),
            BasicBlock(512, 512, 1, None)
        )
        
        # 全局平均池化（自适应输出尺寸为1x1）
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 全连接层（分类器）
        self.fc = nn.Linear(512, num_classes)  # 输入维度为最后通道数512
        
        # 参数初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 使用He初始值
                nn.init.kaiming_normal_(
                    m.weight, 
                    mode='fan_out', 
                    nonlinearity='relu'
                )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

