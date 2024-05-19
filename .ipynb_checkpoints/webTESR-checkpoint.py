import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from torch.optim import Adam
import numpy as np
import cv2
from dataloader import CustomDataset

# 定义模型
class WebSr4Out(nn.Module):
    def __init__(self, input_size, output_size):
        super(WebSr4Out, self).__init__()
        self.fcs = [[nn.Linear(input_size, output_size).cuda() for _ in range(4)] for _ in range(4)]

    def forward(self, x):
        output = [[] for _ in range(4)]
        for i in range(4):
            for j in range(4):
                output[i].append(self.fcs[i][j](x).squeeze(0).squeeze(0))
        return output
    
# class Simple4Out(nn.Module):
#     def __init__(self, input_size, output_size):
#         super(Simple4Out, self).__init__()
#         self.conv1 = nn.Conv2d(1,6,(3,3),(1,1),(1,1))
#         self.conv2 = nn.Conv2d(6,8,(3,3),(1,1),(1,1))
#         self.conv3 = nn.Conv2d(8,input_size,(35,35),(1,1),(0,0))
#         self.flatten = nn.Flatten()
#         self.relu = nn.ReLU()
#         self.fcs = [[nn.Linear(input_size, output_size).cuda() for _ in range(4)] for _ in range(4)]

#     def forward(self, x):
#         x = self.relu(self.conv1(x))
#         x = self.relu(self.conv2(x))
#         x = self.relu(self.conv3(x))
#         x = self.flatten(x)
#         output = [[] for _ in range(4)]
#         for i in range(4):
#             for j in range(4):
#                 output[i].append(torch.sigmoid(self.fcs[i][j](x).squeeze(0).squeeze(0)))
#         return output

# 加载预训练的ResNet-18模型
# model = Simple4Out(16, 1).cuda()
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).cuda()
model.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2, bias=False).cuda()
num_ftrs = model.fc.in_features
model.fc = WebSr4Out(num_ftrs, 1).cuda()

# 定义损失函数和优化器
criterion = nn.BCEWithLogitsLoss().cuda()
optimizer = Adam(model.parameters(), lr=1e-3)

dataset = CustomDataset()
BATCH_SIZE = 2048
train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
test_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=4)               

# 训练模型
for epoch in range(5):
    model.train()
    counter = 0
    for t in range(20):
        for x, y, batch_data, batch_labels in train_dataloader:
            outputs = model(batch_data.cuda())
            loss = 0
            for i in range(4):
                for j in range(4):
                    loss += criterion(outputs[i][j].reshape(batch_labels[i][j].shape).cuda(), batch_labels[i][j].cuda())
            counter += 1
            if counter%100==0:
                print(t, counter,loss)
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()  
            optimizer.step()

    # 在每个epoch结束时评估模型
    model.eval()
    counter = 0
    truth_image = np.zeros((1024, 1024), dtype="uint8")
    output_image = np.zeros((1024, 1024), dtype="uint8")
    with torch.no_grad():
        for bx, by, batch_data, batch_labels in test_dataloader:
            outputs = model(batch_data.cuda())
            counter += 1
            for t in range(len(bx)):    
                for i in range(4):
                    for j in range(4):   
                        if t==0:
                            print(counter*len(bx), outputs[i][j].shape, bx, by)
                        output_image[4*by[t]+i,4*bx[t]+j] = outputs[j][i][t]*255
                        truth_image[4*by[t]+i,4*bx[t]+j] = batch_labels[j][i][t]*255
    cv2.imwrite(f'output/data_sr_{epoch}.bmp', output_image)
    cv2.imwrite(f'output/data_sr_{epoch}_truth.bmp', truth_image)
    print("output")
    
    # 保存模型
    torch.save(model.state_dict(), 'resnet18_binary_classification.pth')
