'''
Description: 
This script is designed for evaluating the performance of various segmentation models on a given dataset.
It includes functionality for loading datasets, selecting models, and calculating evaluation metrics such as Dice coefficient, ASSD, and HD95.
The script also includes a visualization component to display original images, ground truth masks, and predicted masks.

Author: LYD
Date: 2024-11-27 21:08:45
'''
import torch
from torch.utils.data import DataLoader
from load_dataset import SegmentationDataset
from deeplabv3 import DeepLabV3
from unet import UNet
from fcn8s import FCN8s
from vit_seg_modeling import VisionTransformer,CONFIGS
import os
import matplotlib.pyplot as plt
import numpy as np
from medpy.metric import binary

# 计算 Dice 系数
def dice_coefficient(predicted, ground_truth):
    smooth = 1e-6
    # 使用 numpy 进行计算
    pred_class = (predicted == 1).astype(float)
    gt_class = (ground_truth == 1).astype(float)
    intersection = (pred_class * gt_class).sum()
    union = pred_class.sum() + gt_class.sum()
    return (2. * intersection + smooth) / (union + smooth)

# 选择模型函数
def select_model(model_name, device, n_channels, n_classes, pretrained_path=None):
    # 选择模型架构
    if model_name == 'unet':
        model = UNet(n_channels=n_channels, n_classes=n_classes).to(device)
    elif model_name == 'deeplab':
        model = DeepLabV3(n_channels=n_channels, n_classes=n_classes).to(device)
    elif model_name == 'fcn8s':
        model = FCN8s(n_channels=n_channels, n_classes=n_classes).to(device)
    elif model_name == 'transunet':
        config = CONFIGS[vit_model_config]
        model = VisionTransformer(config, img_size=256, num_classes=n_classes, zero_head=False, vis=False).to(device)
    else:
        raise ValueError("无效的模型名称")

    # 如果提供了预训练模型的路径，加载模型权重
    if pretrained_path is not None:
        if os.path.exists(pretrained_path):
            # 加载预训练模型的权重
            checkpoint = torch.load(pretrained_path, map_location=device)
            model.load_state_dict(checkpoint)
            print(f"成功加载训练模型：{pretrained_path}")
        else:
            print(f"警告: 找不到训练模型文件 {pretrained_path}!")

    return model

# 评估函数
def evaluate(model, test_loader, device='cuda', save_dir = f'visualizations'):

    os.makedirs(save_dir, exist_ok=True)

    # 用来存储每个批次的指标
    avg_dice_score = []
    avg_assd_score = []
    avg_hd95_score = []

    # 不计算梯度，减少内存和计算资源的使用
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(test_loader):

            # 移动到GPU
            images = images.to(device)
            masks = masks.to(device)

            # 模型预测
            outputs = model(images)
            outputs = torch.sigmoid(outputs)  # 将输出映射到 [0, 1]
            
            # 二值化处理
            outputs = (outputs > 0.4).float()
            masks = (masks > 0).float()

            # pytorch 张量移到 CPU, 去除通道信息
            outputs_np = outputs.cpu().numpy().squeeze()
            masks_np = masks.cpu().numpy().squeeze()

            # 计算指标
            # 计算 Dice 系数
            dice_score = dice_coefficient(outputs_np, masks_np)

            # 计算 ASSD
            assd_score = binary.assd(outputs_np, masks_np)

            # 计算 HD95
            hd95_score = binary.hd95(outputs_np, masks_np)

            # 将每个批次的结果添加到列表
            avg_dice_score.append(dice_score)
            avg_assd_score.append(assd_score)
            avg_hd95_score.append(hd95_score)

        # 可视化并保存分割结果
        for i in range(images.size(0)):  # 对每个图像进行处理
            # 获取原始图像和预测的分割结果
            img = images[i].cpu().numpy().squeeze()  # 去掉通道维度

            # 确保 img 是二维数组
            if img.ndim == 3 and img.shape[0] == 1:
                img = img[0]

            pred_mask = outputs[i].cpu().numpy().squeeze()  # 去掉通道维度
            if pred_mask.ndim == 3 and pred_mask.shape[0] == 1:
                pred_mask = pred_mask[0]
            
            # 创建子图并展示图像和分割结果
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(img, cmap='gray')
            axes[0].set_title('orginal image')
            axes[1].imshow(masks[i].cpu().numpy().squeeze(), cmap='gray')
            axes[1].set_title('label')
            axes[2].imshow(pred_mask, cmap='gray')
            axes[2].set_title('predict')
            for ax in axes:
                ax.axis('off')

            # 保存图像到指定文件夹
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'pred_{batch_idx * len(images) + i}.png'))
            plt.close(fig)

    # 计算平均指标
    mean_dice_score = np.mean(avg_dice_score)
    mean_assd_score = np.mean(avg_assd_score)
    mean_hd95_score = np.mean(avg_hd95_score)

    # 输出平均值
    print(f'平均 Dice 系数: {mean_dice_score:.4f}')
    print(f'平均 ASSD: {mean_assd_score:.4f}')
    print(f'平均 HD95: {mean_hd95_score:.4f}')

    return mean_dice_score, mean_assd_score, mean_hd95_score

# 获取脚本所在的相对路径
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))  # 相对路径基准

# 数据集路径
is_windows = os.name == 'nt'  # Windows系统
is_linux = os.name == 'posix'  # Linux系统

if is_windows:
    test_img_dir = os.path.join(base_path, r'project1\FedICRA\data\FAZ\Domain3\test\imgs')
    test_mask_dir = os.path.join(base_path, r'project1\FedICRA\data\FAZ\Domain3\test\mask')
elif is_linux:
    test_img_dir = os.path.join(base_path, 'project1/FedICRA/data/FAZ/Domain3/test/imgs')
    test_mask_dir = os.path.join(base_path, 'project1/FedICRA/data/FAZ/Domain3/test/mask')

# 创建测试数据集实例
test_dataset = SegmentationDataset(test_img_dir, test_mask_dir)

# 创建数据加载器
test_loader = DataLoader(test_dataset, batch_size=50, shuffle=False)

# 选择模型名称
model_name = 'transunet'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设置模型的输入输出通道数
n_channels = 1  # 根据数据集修改
n_classes = 1   # 根据数据集修改

#vit model类型
vit_model_config = 'R50-ViT-L_16' # 可选 R50-ViT-B_16 R50-ViT-L_16

# 加载模型
model = select_model(model_name=model_name, device=device, n_channels=n_channels, n_classes=n_classes, pretrained_path = "transunet_model(RL).pth")

# 执行评估
average_dice = evaluate(model, test_loader, device=device)
