# -*- coding = utf-8 -*-
# @Time : 2021/4/1 11:52 下午
# @Author : Pan
# @Software: PyCharm

#自定义的参数要to(device)
#model要to device
#用 tensor.cpu转成cpu的tensor



import torch
from torch import nn
from torch.optim import SGD

#定义一个device对象
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



#准备数据
x = torch.rand([500,1]).to(device)          #这里要to(device)
y_true = 3*x + 0.8                   #这个0.8 不用to(device)

#定义模型，完成一次前向传播
class MyLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1,1)

    def forward(self,x):
        out = self.linear(x)
        return out


my_linear = MyLinear().to(device)  #模型to.(device)
optimizer = SGD(my_linear.parameters(),0.001)
loss_fn = nn.MSELoss()


#循环，进行梯度下降，参数的更新
for i in range(50000):
    y_predict = my_linear(x)
    loss = loss_fn(y_predict,y_true)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i%100==0:
        params = list(my_linear.parameters())
        print(loss.item(),params[0].item(),params[1].item())