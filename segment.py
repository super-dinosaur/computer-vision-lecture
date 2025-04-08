import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import os
import json

from utils import visualize_results, test_model, draw_loss_curve, get_dataloader, CombinedLoss
from nets import UNetWithResNet
from cfg import PATH_VISUALIZATION_RESULTS

architecture = "UNetWithResNet_CBAM"

def train_model(model, dataloader, dataloader_test, criterion, optimizer, scheduler=None, num_epochs=10, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    print(f"使用设备: {device}")
    print(f"开始训练 - 共 {num_epochs} 个Epoch")
    
    # 记录训练损失
    train_losses = []
    # 记录最佳mIoU和epoch
    best_miou = 0.0
    best_epoch = 0
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        
        # 使用tqdm创建进度条
        progress_bar = tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for images, masks in progress_bar:
            images = images.to(device)
            masks = masks.to(device).squeeze(1).long()  # [N, H, W]
            
            # 前向传播
            outputs, aux_outputs = model(images)
            # 计算损失
            loss = criterion(outputs, masks, aux_outputs)
            
            # 反向传播
            optimizer.zero_grad()  # 清除之前的梯度
            loss.backward()
            optimizer.step()       # 更新参数
            
            # 累加批次损失
            epoch_loss += loss.item()
            batch_count += 1
            
            # 更新进度条显示当前损失
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
        
        # 计算平均损失
        avg_loss = epoch_loss / batch_count
        train_losses.append(avg_loss)
        
        # epoch结束后更新学习率
        if scheduler is not None:
            scheduler.step()
            
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{num_epochs} 完成，平均损失: {avg_loss:.4f}, 学习率: {current_lr:.6f}")
        
        # 每5个epoch评估模型并保存最佳模型
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            print(f"在验证集上评估模型...")
            # 在验证集上测试
            avg_metrics = test_model(model, dataloader_test)
            current_miou = avg_metrics['mIoU']
            print(f"Epoch {epoch+1} 验证集 mIoU: {current_miou:.4f}")
            if current_miou > best_miou:
                best_miou = current_miou
                best_epoch = epoch + 1
    
    print(f"训练完成! 最佳模型在Epoch {best_epoch}, mIoU: {best_miou:.4f}")

    path_loss_curve = os.path.join(PATH_VISUALIZATION_RESULTS, f'train_loss_{num_epochs}_{architecture}.png')
    draw_loss_curve(train_losses, best_miou, best_epoch, num_epochs, architecture, path_loss_curve)
    # save list train_losses
    with open(os.path.join(PATH_VISUALIZATION_RESULTS, f'train_loss_{num_epochs}_{architecture}.json'), 'w') as f: json.dump(train_losses, f)
    return model

if __name__ == "__main__":
    dataloader_train = get_dataloader()
    dataloader_test = get_dataloader(split="val")
    
    # 创建改进版模型
    model = UNetWithResNet(num_classes=21, pretrained=True)
    
    # 组合损失函数
    criterion = CombinedLoss(weights=[0.5, 0.3, 0.2], num_classes=21)
    
    # 使用SGD优化器
    lr = 1e-3
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    
    # 添加学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-5)
    
    # 训练模型
    model = train_model(model, dataloader_train, dataloader_test, criterion, optimizer, scheduler, num_epochs=300, lr=lr)
    
    # 可视化结果
    visualize_results(model, dataloader_train, PATH_VISUALIZATION_RESULTS)
    
    # 最终测试
    print("在最终模型上进行测试...")
    avg_metrics = test_model(model, dataloader_test)