import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from torch.optim import AdamW
import numpy as np
import cv2
from dataloader import CustomDataset
from model import *

BATCH_SIZE = 4096
EPOCH_NUM = 20
WEIGHT_NAME = 'weight/resnet18_sr_x2.pth'
image_path = 'data/data_lr_vari.jpg' 
test_path = 'data/data_lr_vari_test.jpg'  
truth_path = 'data/data_hr_vari.bmp'
test_truth_path = "data/data_hr_vari_test.bmp"


if __name__ == "__main__":
    dataset = CustomDataset(image_path, truth_path)
    test_dataset = CustomDataset(test_path, test_truth_path)
    train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=4)               

    # 加载预训练的ResNet-18模型
    # model = Simple4Out((7,7), (2,2)).cuda()
    model = WebSr4Out((2,2)).cuda()
    # model.load_model(WEIGHT_NAME)

    # 定义损失函数和优化器
    criterion = nn.BCELoss().cuda()
    cross_loss = CrossLoss2x2()
    optimizer = AdamW(model.parameters(), lr=1e-3)

    # 训练模型
    t_counter = 0
    for epoch in range(EPOCH_NUM):
        model.train()
        for x, y, batch_data, batch_labels in train_dataloader:
            batch_data = batch_data.cuda() + 0.001 * torch.randn(batch_data.size()).cuda()
            batch_labels = batch_labels.cuda()
            outputs = model(batch_data).cuda()
            # print(outputs.shape, batch_labels.shape)
            loss = criterion(outputs, batch_labels) + cross_loss(outputs, batch_labels) * 0.01
            t_counter += 1
            if t_counter%10==0:
                print(t_counter,loss)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()  
            optimizer.step()

        # 在每个epoch结束时评估模型
        model.eval()
        counter = 0
        truth_image = np.zeros((test_dataset.truth_size[1], test_dataset.truth_size[1]), dtype="uint8")
        output_image = np.zeros((test_dataset.truth_size[1], test_dataset.truth_size[1]), dtype="uint8")
        with torch.no_grad():
            for bx, by, batch_data, batch_labels in test_dataloader:
                outputs: torch.Tensor = model(batch_data.cuda())
                counter += 1
                for t in range(len(bx)):    
                    if t==0:
                        print(counter*len(bx), outputs.shape, len(bx), bx)
                    sf = test_dataset.super_factor
                    output_image[sf[1]*by[t]:sf[1]*by[t]+sf[1],sf[0]*bx[t]: sf[0]*bx[t]+sf[0]] = outputs[t,0,:,:].clamp(0,1).cpu().detach().numpy()*255
                    # truth_image[4*by[t]:4*by[t]+4,4*bx[t]: 4*bx[t]+4] = batch_labels[t].clamp(0,1).cpu().detach().numpy()*255
        cv2.imwrite(f'output/data_sr_{epoch}_t.jpg', output_image)
        # cv2.imwrite(f'output/data_sr_{epoch}_truth_t.bmp', truth_image)
        print("output")
        
        # 保存模型
        model.save_model(WEIGHT_NAME)
