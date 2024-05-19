import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# 定义数据集类
class CustomDataset(Dataset):
    def __init__(self):
        # 定义滑动窗口的尺寸和步长
        window_size = (35, 35)  # 假设窗口大小为32x32
        stride = 1  # 假设滑动步长为16
        # 定义数据转换
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Pad((17,17,17,17),padding_mode='symmetric')
        ])
        # 读取单张图片
        image_path = 'data/data_lr_vari.jpg'  # 替换为你的图片路径
        truth_path = 'data/data_hr_vari.bmp'
        image = Image.open(image_path)
        truth = transforms.ToTensor()(Image.open(truth_path)).clamp(0,1)
        # 对图片进行预处理
        image_tensor: torch.Tensor = transform(image)

        self.patches = []
        for y in range(0, image_tensor.size()[1] - window_size[0], stride):
            for x in range(0, image_tensor.size()[2] - window_size[1], stride):
                patch = image_tensor[:, y:y+window_size[0], x:x+window_size[1]]
                upsample_answer = [[truth[0,4*y+i,4*x+j] for i in range(4)] for j in range(4)]
                self.patches.append((x, y, patch, upsample_answer))

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        # 这里可以添加你自己的数据加载逻辑
        x, y, data, label = self.patches[idx]
        return x, y, data, label

