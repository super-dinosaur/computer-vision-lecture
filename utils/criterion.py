import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # 将输入转换为概率
        inputs = torch.softmax(inputs, dim=1)
        
        # 将目标转换为one-hot编码
        targets_one_hot = torch.zeros_like(inputs)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)
        
        # 展平预测和目标
        inputs = inputs.view(-1)
        targets_one_hot = targets_one_hot.view(-1)
        
        intersection = (inputs * targets_one_hot).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets_one_hot.sum() + smooth)
        
        return 1 - dice

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, weight=None, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.ce_fn = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index)
        
    def forward(self, inputs, targets):
        logpt = -self.ce_fn(inputs, targets)
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * self.alpha * logpt
        return loss

class CombinedLoss(nn.Module):
    def __init__(self, weights=[0.5, 0.3, 0.2], num_classes=21):
        super(CombinedLoss, self).__init__()
        self.weights = weights
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=255)
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(gamma=2)
        
    def forward(self, inputs, targets, aux_inputs=None):
        if aux_inputs is not None:  # 使用深度监督
            ce_main = self.ce_loss(inputs, targets)
            dice_main = self.dice_loss(inputs, targets)
            focal_main = self.focal_loss(inputs, targets)
            
            ce_aux = self.ce_loss(aux_inputs, targets)
            dice_aux = self.dice_loss(aux_inputs, targets)
            focal_aux = self.focal_loss(aux_inputs, targets)
            
            # 主要输出的损失
            main_loss = self.weights[0] * ce_main + self.weights[1] * dice_main + self.weights[2] * focal_main
            # 辅助输出的损失 (权重较低)
            aux_loss = self.weights[0] * ce_aux + self.weights[1] * dice_aux + self.weights[2] * focal_aux
            
            # 总损失
            return main_loss + 0.4 * aux_loss
        else:  # 没有深度监督
            ce = self.ce_loss(inputs, targets)
            dice = self.dice_loss(inputs, targets)
            focal = self.focal_loss(inputs, targets)
            return self.weights[0] * ce + self.weights[1] * dice + self.weights[2] * focal