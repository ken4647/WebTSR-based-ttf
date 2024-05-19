import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

Image.MAX_IMAGE_PIXELS = None

# 定义数据集类
class CustomDataset(Dataset):
    def __init__(self, image_path, truth_path, window_size: tuple[int]=(13,13), stride: int=1):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Pad((window_size[0]//2,window_size[1]//2,window_size[0]//2,window_size[1]//2),padding_mode='symmetric'),
        ])
        self.truth = transforms.ToTensor()(Image.open(truth_path)).clamp(0,1)
        self.image_tensor: torch.Tensor = transform(Image.open(image_path))

        self.window_size = window_size
        self.stride = stride
        
        self.image_size = (self.image_tensor.size()[1]-self.window_size[1]//2*2, self.image_tensor.size()[2]-self.window_size[1]//2*2)
        self.truth_size = self.truth.size()
        self.super_factor = (int(self.truth.size()[1]/(self.image_tensor.size()[1]-window_size[0]//2*2)), int(self.truth.size()[2]/(self.image_tensor.size()[2]-window_size[1]//2*2)))

    def __len__(self):
        return self.image_size[0]*self.image_size[1]//self.stride

    def __getitem__(self, idx):
        sf = self.super_factor
        x = (idx*self.stride)%(self.image_tensor.size()[1]-self.window_size[0]//2*2)
        y = (idx*self.stride)//(self.image_tensor.size()[1]-self.window_size[0]//2*2)
        patch = self.image_tensor[:, y:y+self.window_size[0], x:x+self.window_size[1]]
        upsample_answer = self.truth[:,sf[0]*y:sf[0]*y+sf[0], sf[1]*x:sf[1]*x+sf[1]]
        # print((self.image_tensor.size()[1]-self.window_size[0]//2*2), patch.shape, upsample_answer.shape, idx//4096, x, y)
        return x, y, patch, upsample_answer

