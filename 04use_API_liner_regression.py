# -*- coding = utf-8 -*-
# @Time : 2021/3/29 11:37 下午
# @Author : Pan
# @Software: PyCharm

#nn.Module类,要继承父类的init方法（用super)然后self.liner = nn.Linear(1,1) (输入输出的特征数量！！),还要必须实现forward方法(完成一次前向计算).
#优化器类 optimizer = optim.SGD(model.parameters(),lr=0.01)
#交叉商 nn.CrossEntropyLose()

import torch
from torch import nn
from torch.optim import SGD

#准备数据
x = torch.rand([500,1])
y_true = 3*x + 0.8

#定义模型，完成一次前向传播
class MyLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1,1)

    def forward(self,x):
        out = self.linear(x)
        return out


#实例化模型，优化器类实例化，loss实例化
my_linear = MyLinear()
optimizer = SGD(my_linear.parameters(),0.001)  #对象.方法       nn.Module里有parameters()构造方法
loss_fn = nn.MSELoss()


#循环，进行梯度下降，参数的更新
for i in range(50000):
    y_predict = my_linear(x)          #这里返回的显然是上面的out
    loss = loss_fn(y_predict,y_true)
    optimizer.zero_grad()             #梯度置0
    loss.backward()                   #损失反传
    optimizer.step()            #参数更新

    if i%100==0:
        params = list(my_linear.parameters())
        print(loss.item(),params[0].item(),params[1].item())