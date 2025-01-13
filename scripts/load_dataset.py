import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class SegmentationDataset(Dataset):
    def __init__(self, img_dirs, mask_dir, transform=None):
        self.img_dirs = [os.path.abspath(d) for d in img_dirs] if isinstance(img_dirs, list) else [os.path.abspath(img_dirs)]
        self.mask_dir = os.path.abspath(mask_dir)
        self.transform = transform
        # 构建图像和标签的配对列表
        self.image_label_pairs = self._build_pairs()

    def _build_pairs(self):
        pairs = []
        for img_dir in self.img_dirs:  # 遍历每个图像目录
            for img_name in sorted(os.listdir(img_dir)):  # 遍历目录中的图像文件
                # 去掉图像文件名中的 "_fake"
                base_name = img_name.replace("_fake", "")
                img_path = os.path.join(img_dir, img_name)  # 完整图像路径
                mask_path = os.path.join(self.mask_dir, base_name)  # 标签路径
                
                if os.path.exists(img_path) and os.path.exists(mask_path):
                    pairs.append((img_path, mask_path))
                else:
                    print(f"文件未找到: {img_path} 或 {mask_path}")
        return pairs

    def __len__(self):
        return len(self.image_label_pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.image_label_pairs[idx]

        # 检查路径是否存在
        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            raise FileNotFoundError(f"图像或标签文件不存在: {img_path}, {mask_path}")

        # 打开图像和标签
        image = Image.open(img_path).convert("L")  # 假设图像是RGB格式
        mask = Image.open(mask_path).convert("L")  # 标签是灰度图

        # 如果有变换，应用到图像
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        # 将标签转为Tensor，二值化
        mask = transforms.ToTensor()(mask)  # 结果为[C, H, W]，对于灰度图C=1
        mask = (mask > 0).float()  # 二值化

        return image, mask