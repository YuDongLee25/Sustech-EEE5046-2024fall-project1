'''
Description: This script is designed for training various segmentation models using PyTorch. 
It includes functionality for data loading, model selection, training, validation, and learning rate scheduling. 
The script also features an experimental integration with a DQN agent for dynamic learning rate adjustment.

Author: LYD
Date: 2024-11-26 16:33:38
'''
import os
import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import argparse
from torch.utils.data import DataLoader
from torchvision import transforms
from load_dataset import SegmentationDataset
from transunet import TransUNet
from fcn8s import FCN8s
from unet import UNet
from deeplabv3 import DeepLabV3
from torch.optim.lr_scheduler import CosineAnnealingLR
from dqn_learning_rate import DQNAgent
from vit_seg_modeling import VisionTransformer,CONFIGS

# 创建一个 xlsx 来保存每个 epoch 的数据
columns = ['Epoch', 'Learning Rate']
log_file = "learning_rate_log.xlsx"

# 命令行参数解析
def parse_args():
    parser = argparse.ArgumentParser(description="Segmentation Model Training")

    # 超参数
    parser.add_argument('--num_epochs', type=int, default=300, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='Initial learning rate')
    parser.add_argument('--optimizer', type=str, default='AdamW', choices=['AdamW', 'Adam', 'SGD'], help='Optimizer choice')
    parser.add_argument('--loss_function', type=str, default='BCEWithLogitsLoss', choices=['BCEWithLogitsLoss'], help='Loss function')
    parser.add_argument('--n_channels', type=int, default=1, help='Number of input channels')
    parser.add_argument('--n_classes', type=int, default=1, help='Number of output classes')
    parser.add_argument('--cuda', type=bool, default=True, help='Use GPU if available')
    parser.add_argument('--model', type=str, default='transunet', choices=['unet', 'deeplab', 'transunet', 'fcn8s', 'vit'], help='Model type')

    return parser.parse_args()

# 获取命令行参数
args = parse_args()

# 获取脚本所在的相对路径
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))  # 相对路径基准

# 所有数据集路径
is_windows = os.name == 'nt'  # Windows系统
is_linux = os.name == 'posix'  # Linux系统

if is_windows:
    test_img_dir = os.path.join(base_path, r'project1\FedICRA\data\FAZ\Domain2\train\imgs')
    test_mask_dir = os.path.join(base_path, r'project1\FedICRA\data\FAZ\Domain2\train\mask')
    original_img_dir = os.path.join(base_path, r'project1\FedICRA\data\FAZ\Domain1\test\imgs')
    translated_img_dirs = [
        os.path.join(base_path, r'project1\FedICRA\data\FAZ\Domain1_2\train\imgs'),
        os.path.join(base_path, r'project1\FedICRA\data\FAZ\Domain1_3\train\imgs'),
        os.path.join(base_path, r'project1\FedICRA\data\FAZ\Domain1_4\train\imgs'),
    ]
    mask_dir = os.path.join(base_path, r'project1\FedICRA\data\FAZ\Domain1\test\mask')
    # 将原始图像文件夹和领域迁移后的图像文件夹合并
    img_dirs = [original_img_dir] + translated_img_dirs

elif is_linux:
    test_img_dir = os.path.join(base_path, 'project1/FedICRA/data/FAZ/Domain2/train/imgs')
    test_mask_dir = os.path.join(base_path, 'project1/FedICRA/data/FAZ/Domain2/train/mask')
    original_img_dir = os.path.join(base_path, 'project1/FedICRA/data/FAZ/Domain1/train/imgs')
    translated_img_dirs = [
        os.path.join(base_path, 'project1/FedICRA/data/FAZ/Domain1_2/train/imgs'),
        os.path.join(base_path, 'project1/FedICRA/data/FAZ/Domain1_3/train/imgs'),
        os.path.join(base_path, 'project1/FedICRA/data/FAZ/Domain1_4/train/imgs'),
    ]
    mask_dir = os.path.join(base_path, 'project1/FedICRA/data/FAZ/Domain1/test/mask')
    # 将原始图像文件夹和领域迁移后的图像文件夹合并
    img_dirs = [original_img_dir] + translated_img_dirs

# 创建验证数据集实例
val_dataset = SegmentationDataset(img_dirs=test_img_dir, mask_dir=test_mask_dir)
val_loader = DataLoader(val_dataset, batch_size=10, shuffle=True)

# 检查GPU
device = torch.device("cuda" if (torch.cuda.is_available() and args.cuda) else "cpu")
print(f"Using device: {device}")

# 数据增强
def add_noise(image):
    noise = torch.randn_like(image) * torch.rand(1)  # 随机正态分布噪声
    image = image + noise
    image = torch.clamp(image, 0, 1)  # 确保图像值在[0, 1]范围内
    return image

transform = transforms.Compose([
    transforms.RandomApply([transforms.RandomHorizontalFlip()], p=0.5),
    transforms.RandomApply([transforms.RandomVerticalFlip()], p=0.5),
    transforms.RandomApply([transforms.RandomRotation(degrees=(-30, 30))], p=0.5),
    transforms.RandomApply([transforms.RandomAffine(0, translate=(0.1, 0.1))], p=0.5),
    transforms.Resize((256, 256)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.RandomApply([transforms.Lambda(lambda x: add_noise(x))], p=0.5),
])

# 创建训练数据集实例
train_dataset = SegmentationDataset(img_dirs=img_dirs, mask_dir=mask_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

# 选择训练模型
def select_model(model_name):
    if model_name == 'unet':
        return UNet(n_channels=args.n_channels, n_classes=args.n_classes).to(device)
    elif model_name == 'deeplab':
        return DeepLabV3(n_channels=args.n_channels,n_classes=args.n_classes).to(device)
    elif model_name == 'vit':
        return TransUNet(in_channels=args.n_channels, out_channels=args.n_classes).to(device)
    elif model_name == 'fcn8s':
        return FCN8s(n_channels=args.n_channels, n_classes=args.n_classes).to(device)
    elif model_name == 'transunet':
        config = CONFIGS['R50-ViT-L_16']
        return VisionTransformer(config, img_size=256, num_classes=args.n_classes, zero_head=False, vis=False).to(device)
    else:
        raise ValueError("Invalid model name")

model = select_model(args.model)
print(f"Using {args.model} model for training")
model.train()

# 选择损失函数
criterion = nn.BCEWithLogitsLoss()

# 初始化优化器
if args.optimizer == 'AdamW':
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
elif args.optimizer == 'Adam':
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
elif args.optimizer == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)

# 余弦退火学习率调度器
scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-6)

# 计算dice系数
def dice_coefficient(predicted, ground_truth):
    smooth = 1e-6
    # 计算交集和并集
    intersection = (predicted * ground_truth).sum()
    union = predicted.sum() + ground_truth.sum()
    # 返回 Dice 系数
    return (2. * intersection + smooth) / (union + smooth)

# 训练函数
def train_epoch(epoch, model, train_loader, optimizer, criterion):
    
    model.train()
    
    epoch_loss = 0.0
    dice_sum = 0.0
    
    for batch_idx, (images, masks) in enumerate(train_loader):
        
        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        loss = criterion(outputs, masks.float())
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        # 模型预测
        outputs = torch.sigmoid(outputs)  # 将输出映射到 [0, 1]
        
        # 二值化处理
        outputs = (outputs > 0.5).float()
        masks = (masks > 0).float()
        
        # 计算指标
        # 计算本批次Dice系数
        dice_sum += dice_coefficient(outputs, masks)

    avg_loss = epoch_loss / len(train_loader)
    avg_dice = dice_sum / len(train_loader)

    return avg_loss, avg_dice

global pre_val_loss
pre_val_loss = 0.0
# 验证函数
def validate_epoch(epoch, model, val_loader, criterion):
    
    model.eval()
    
    val_loss = 0.0
    dice_sum = 0.0
    pre_val_dice = 0.0
    
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(val_loader):
            # 加载图像和标签数据
            images, masks = images.to(device), masks.to(device)
            # 模型输出
            outputs = model(images)
            # 损失计算
            loss = criterion(outputs, masks.float())
            # 该批次损失
            val_loss += loss.item()
            # 模型预测
            outputs = torch.sigmoid(outputs)
            # 二值化处理
            outputs = (outputs > 0.5).float()
            masks = (masks > 0).float()
            # 计算指标
            # 计算本批次Dice系数
            dice_sum += dice_coefficient(outputs, masks)
            
    avg_val_loss = val_loss / len(val_loader)
    avg_dice = dice_sum / len(val_loader)
    
    # 评估过程使用 DQN 智能体调整学习率
    #agent.adjust_learning_rate(reward=(0.7 * (avg_dice - pre_val_dice) + 0.3 * (pre_val_loss - avg_val_loss) if avg_dice >= 0.01 and avg_val_loss <= 0.2 else -1), next_state = avg_dice - 0.5 * avg_val_loss, optimizer=optimizer)
    agent.adjust_learning_rate(reward= pre_val_loss - avg_val_loss if avg_val_loss <= 0.2 else -1, next_state = avg_val_loss, optimizer=optimizer)

    pre_val_dice = avg_dice
    
    return avg_val_loss, avg_dice

# 智能体实例
agent = DQNAgent(input_dim=1, output_dim=3, lr = args.learning_rate, device=device)

for epoch in range(1, args.num_epochs + 1):

    print(f"Epoch [{epoch}/{args.num_epochs}]")

    # 训练
    train_loss, train_dice = train_epoch(epoch, model, train_loader, optimizer, criterion)
    print(f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}")

    # 验证
    val_loss, val_dice = validate_epoch(epoch, model, val_loader, criterion)
    print(f"Validation Loss: {val_loss:.4f}, Validation Dice: {val_dice:.4f}")
    if is_linux:
        # 保存学习率到 Excel
        learning_rate = optimizer.param_groups[0]['lr']
        existing_data = pd.read_excel(log_file)
        # 创建新的数据行
        new_data = pd.DataFrame({f"epoch_{args.model}": [epoch], f"learning_rate_{args.model}": [learning_rate]})
        # 将新的数据行追加到现有的数据中
        data = pd.concat([existing_data, new_data], ignore_index=True)
        # 将更新后的数据写回到Excel文件
        data.to_excel(log_file, index=False)
    
    # 学习率余弦退火
    #scheduler.step()

    # 保存最佳模型
    if val_loss > pre_val_loss:
        torch.save(model.state_dict(), f"best_{args.model}_model.pth")
    pre_val_loss = val_loss
    
    model_path = f"{args.model}_model.pth"
    torch.save(model.state_dict(), model_path)