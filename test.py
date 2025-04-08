import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights, ResNet152_Weights
from nets import DoubleConv

class UNetWithResNet(nn.Module):
    def __init__(self, num_classes=21, pretrained=True):
        super(UNetWithResNet, self).__init__()

        # 加载预训练的ResNet50模型作为编码器
        #resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        resnet = models.resnet152(weights=ResNet152_Weights.IMAGENET1K_V1 if pretrained else None)
        
        # 编码器部分
        self.firstconv = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.enc1 = resnet.layer1  # 输出通道: 256
        self.enc2 = resnet.layer2  # 输出通道: 512
        self.enc3 = resnet.layer3  # 输出通道: 1024
        self.enc4 = resnet.layer4  # 输出通道: 2048
        
        # 解码器部分
        self.up5 = nn.ConvTranspose2d(2048, 1024, 2, stride=2)
        self.dec5 = DoubleConv(2048, 1024)  # 1024 + 1024 = 2048
        
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec6 = DoubleConv(1024, 512)   # 512 + 512 = 1024
        
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec7 = DoubleConv(512, 256)    # 256 + 256 = 512
        
        self.up8 = nn.ConvTranspose2d(256, 64, 2, stride=2)
        self.dec8 = DoubleConv(128, 64)     # 64 + 64 = 128
        
        # 添加空洞卷积以扩大感受野
        self.dilated_conv = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # 分类层
        self.final = nn.Conv2d(64, num_classes, 1)
        
        # 深度监督
        self.aux_classifier = nn.Conv2d(256, num_classes, 1)

    def forward(self, x):
        # 保存原始输入尺寸用于最终上采样
        input_size = x.size()[2:]
        
        # 初始下采样
        x = self.firstconv(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)(x)
        
        # 编码器部分
        e1 = self.enc1(x)      # 256 channels
        e2 = self.enc2(e1)     # 512 channels
        e3 = self.enc3(e2)     # 1024 channels
        e4 = self.enc4(e3)     # 2048 channels
        
        # 解码器部分
        d5 = self.up5(e4)
        d5 = torch.cat([d5, e3], dim=1)
        d5 = self.dec5(d5)
        
        d6 = self.up6(d5)
        d6 = torch.cat([d6, e2], dim=1)
        d6 = self.dec6(d6)
        
        d7 = self.up7(d6)
        d7 = torch.cat([d7, e1], dim=1)
        d7 = self.dec7(d7)
        
        # 辅助输出用于深度监督
        aux_out = self.aux_classifier(d7)
        # 将辅助输出上采样到原始输入尺寸
        aux_out = nn.functional.interpolate(aux_out, size=input_size, mode='bilinear', align_corners=True)
        
        d8 = self.up8(d7)
        # 调整大小以匹配初始特征图
        d8 = nn.functional.interpolate(d8, size=x.shape[2:], mode='bilinear', align_corners=True)
        d8 = torch.cat([d8, x], dim=1)
        d8 = self.dec8(d8)
        
        # 应用空洞卷积
        d8 = self.dilated_conv(d8)
        
        # 最终分类层
        output = self.final(d8)
        
        # 将输出上采样到原始输入尺寸
        output = nn.functional.interpolate(output, size=input_size, mode='bilinear', align_corners=True)
        
        if self.training:
            return output, aux_out
        else:
            return output
        
if __name__ == "__main__":
    # Examples::
    # With square kernels and equal stride
    m = nn.ConvTranspose2d(16, 33, 3, stride=2)
    # non-square kernels and unequal stride and with padding
    m = nn.ConvTranspose2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
    input = torch.randn(20, 16, 50, 100)
    output = m(input)
    # exact output size can be also specified as an argument
    input = torch.randn(1, 16, 12, 12)
    downsample = nn.Conv2d(16, 16, 3, stride=2, padding=1)
    upsample = nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1)
    h = downsample(input)
    h.size()
    #torch.Size([1, 16, 6, 6])
    output = upsample(h, output_size=input.size())
    output.size()
    #torch.Size([1, 16, 12, 12])
