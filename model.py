import torch.nn as nn
from kan import KAN, KANLinear
from torchvision import models
import torch


# 定义模型
class WebSr4Out(nn.Module):
    def __init__(self, output_size):
        super(WebSr4Out, self).__init__()
        self.output_size = output_size
        
        self.model = models.resnet18(weights=None)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2, bias=False)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, output_size[0]*output_size[1])
        self.sig = nn.Sigmoid()

    def forward(self, x):
        return self.sig(self.model(x).view((-1, 1, self.output_size[0], self.output_size[1])))
    
    def load_model(self, weight_path:str):
        self.model.load_state_dict(torch.load(weight_path))
        
    def save_model(self, weight_path:str):
        torch.save(self.model.state_dict(), weight_path)
        
    
class Simple4Out(nn.Module):
    def __init__(self, input_size, output_size):
        super(Simple4Out, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.conv1 = nn.Conv2d(1,6,(3,3),(1,1),(1,1))
        self.conv2 = nn.Conv2d(6,8,(3,3),(1,1),(1,1))
        self.conv3 = nn.Conv2d(8,16,self.input_size,(1,1),(0,0))
        self.fcs = KANLinear(16, self.output_size[0]*self.output_size[1])
        self.flatten = nn.Flatten()
        self.relu = nn.SiLU()
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.flatten(x)
        return self.sig(self.fcs(x).view((-1, 1, self.output_size[0], self.output_size[1])))
    
class Simple4Out(nn.Module):
    def __init__(self, input_size, output_size):
        super(Simple4Out, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.conv1 = nn.Conv2d(1,6,(3,3),(1,1),(1,1))
        self.conv2 = nn.Conv2d(6,8,(3,3),(1,1),(1,1))
        self.conv3 = nn.Conv2d(8,16,self.input_size,(1,1),(0,0))
        self.fcs = nn.Linear(16, self.output_size[0]*self.output_size[1])
        self.flatten = nn.Flatten()
        self.relu = nn.LeakyReLU()
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.flatten(x)
        return self.sig(self.fcs(x).view((-1, 1, self.output_size[0], self.output_size[1])))
    
class CrossLoss2x2(nn.Module):
    def __init__(self):
        super(CrossLoss2x2, self).__init__()
        
    def forward(self, x:torch.Tensor, y:torch.Tensor):
        return (((x[:,:,0, 0]+x[:,:,1, 1])-(y[:,:,0,0]+y[:,:,1,1]))**2+((x[:,:,0, 1]+x[:,:,1, 0])-(y[:,:,0,1]+y[:,:,1,0]))**2).mean()
