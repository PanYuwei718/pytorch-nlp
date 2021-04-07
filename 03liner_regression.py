# -*- coding = utf-8 -*-
# @Time : 2021/3/28 3:45 下午
# @Author : Pan
# @Software: PyCharm

import torch
from matplotlib import pyplot as plt
#requires_grad置True时，该tensor后续的操作都会被记录在grad_fn里面. 置True时，要用数据用tensor.data，不然会累加梯度
#requires_grad置True时，不能直接.numpy() 要用tensor.detach().numpy()
#output.backward()反向传播。tensor.grad 获取梯度（获取的是累加的梯度） 每次反向传播之前需要先把梯度置0再反向传播


#准备数据.给出满足y=3x+0.8的一些
x = torch.rand([500,1])
y_true = 3*x + 0.8


#给个学习率
learning_rate = 0.1

#通过模型计算y_predict
w = torch.rand([1,1],requires_grad=True)
b = torch.tensor([0],requires_grad=True,dtype=torch.float32)


#循环反响传播，更新参数
for i in range(100):
    y_predict = torch.matmul(x,w)+b         #y_predict, loss要放里面
    loss = (y_true - y_predict).pow(2).mean()
    if w.grad is not None:
        w.grad.data.zero_()
    if b.grad is not None:
        b.grad.data.zero_()

    #计算loss
    loss.backward()
    w.data = w.data - learning_rate*w.grad
    b.data = b.data - learning_rate*b.grad
    print("w,b,loss",w.item(),b.item(),loss.item())

plt.figure(figsize=(20,8))
plt.scatter(x.reshape(-1),y_true.reshape(-1)) #给的数据的点，满足y=3x+0.8的散点图

y_predict = torch.matmul(x,w) + b
plt.plot(x.reshape(-1),y_predict.detach().numpy().reshape(-1)) #循环结束后的最终参数的直线图

plt.show()