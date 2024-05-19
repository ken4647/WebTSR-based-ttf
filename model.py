import torch.nn as nn
from kan import KAN, KANLinear


# 定义模型
class WebSr4Out(nn.Module):
    def __init__(self, input_size, output_size):
        super(WebSr4Out, self).__init__()
        self.output_size = output_size
        self.sig = nn.Sigmoid()
        self.fcs = nn.Linear(input_size, output_size[0]*output_size[1])

    def forward(self, x):
        return self.sig(self.fcs(x).view((-1, 1, self.output_size[0], self.output_size[1])))
    
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