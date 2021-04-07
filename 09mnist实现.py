# -*- coding = utf-8 -*-
# @Time : 2021/4/4 10:09 下午
# @Author : Pan
# @Software: PyCharm

import os
import torch
from torchvision.datasets import MNIST
from torchvision.transforms import Compose,ToTensor,Normalize
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch import nn
import torch.nn.functional as F
import numpy as np

BATCH_SIZE=128
TEST_BATCH_SIZE=1000

#准备数据集
def get_dataloader(train=True,batch_size=BATCH_SIZE):
    transform_fn = Compose([
        ToTensor(),
        Normalize(mean=(0.1307,),std=(0.3081,))
    ])
    dataset = MNIST(root="./data",train=train,transform=transform_fn)
    data_loader = DataLoader(dataset,batch_size=batch_size,shuffle=True)
    return data_loader


#构建模型,mnist是 1*28*28 （通道数*长*宽）
class MnistModel(nn.Module):
    def __init__(self):
        super(MnistModel, self).__init__()
        self.fc1 = nn.Linear(1*28*28,100)
        self.fc2 = nn.Linear(100,10)

    def forward(self,input):   #input: [batch_size,1,28,28]
        x = input.view([-1,1*28*28]) #或者 x = input.view([input.size(0),1*28*28]) 因为input已经知道了

        #进行全连接的操作
        x = self.fc1(x)
        x = F.relu(x)
        out = self.fc2(x)
        return out


#开始训练： 实例化模型，实例化优化器类，获取遍历dataloader，梯度置0，进行前向计算，计算损失，反向传播，更新参数

criterion = nn.CrossEntropyLoss()
model = MnistModel()
optimizer = Adam(model.parameters(),lr=0.001)

#加载保存好的模型。（一上来损失就很小。而且训练速度快。）
if os.path.exists("./model/model.pkl"):                             #如果路径存在
    model.load_state_dict(torch.load("./model/model.pkl"))          #加载模型
    optimizer.load_state_dict(torch.load("./model/optimizer.pkl"))  #加载优化器

def train(epoch):
    criterion = nn.CrossEntropyLoss()  #里面自带了softmax
    data_loader = get_dataloader()                           #这里不带参数，表示按照原函数给的来。
    for idx,(input,target) in enumerate(data_loader):
        optimizer.zero_grad()  #梯度置0
        output = model(input)  #调用模型，得到预测值
        loss = criterion(output,target)
        loss.backward()
        optimizer.step()
        if idx%10 == 0:
        #print(epoch,idx,loss.item())
            print('Train Epoch: {} [{}/{} ({:.0f}%)]  Loss:{:.6f}'.format(
            epoch, idx*len(input), len(data_loader.dataset),
            100* idx / len(data_loader), loss.item()))       #len(data_loader),469,表示一个epoch要进行469个batch

        #模型保存
        if idx%100 == 0:
            torch.save(model.state_dict(),"./model/model.pkl")
            torch.save(optimizer.state_dict(),"./model/optimizer.pkl")

#模型的评估
def test():
    loss_list= []
    acc_list = []
    test_dataloader = get_dataloader(train=False,batch_size=TEST_BATCH_SIZE)        #函数的重载
    for idx,(input,target) in enumerate (test_dataloader):
        with torch.no_grad():   #因为不是对计算进行追踪
            output = model(input)
            cur_loss  = F.nll_loss(output,target)  #nll_loss softmax，取对数，去掉负号，算均值。
            loss_list.append(cur_loss)

            #计算准确率
            pred = output.max(dim= -1)[-1]    #获取预测值： 获取最大值，和最大值的位置
            cur_acc= pred.eq(target).float().mean() #获取准确率：先比较，布尔类型。 再转换成浮点，再取均值。
            acc_list.append(cur_acc)
    print("平均准确率，平均损失",np.mean(acc_list),np.mean(loss_list))

if __name__ == '__main__':
    for i in range(1):      #训练三个epoch
        train(i)
    test()


