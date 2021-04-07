# -*- coding = utf-8 -*-
# @Time : 2021/3/27 11:45 下午
# @Author : Pan
# @Software: PyCharm

import torch
import numpy as np

#用列表或序列创建

a = torch.tensor([[1.,-1.],[1.,-1.]])
b = torch.tensor([1,2,3])

print(a)
print(type(a))
print(b)
print(b.dtype)  #torch.int64,以前是自动转换成float的好像。。


#numpy数组创建
array1 = np.arange(0,12).reshape(3,4)
print(torch.tensor(array1))

#创建空数组
print(torch.empty(3,4))

print("___________1111111_______________")



#zeros,ones,rand,randint,randn
print(torch.rand(2,3))  #创建数值在0-1的tensor

print(torch.randint(low=0,high=3,size=[3,4]))

print(torch.randn(2,3)) #正态分布的tensor

print("___________2222222___________")


#张量的方法和属性, 几对[]表示几维的tensor
print(array1.item(1,1))  #给坐标,取出array1中的元素

print(a.numpy()) #tensor转numpy数组

print(a.size())  #torch.Size([2, 2])

print("___________3333333___________")


#修改形状,-1表示根据参数自己变。 这和转置可不一样。。。。。。
print(a.view([1,4])) #tensor([ 1., -1.,  1., -1.])
print(a.view([4,-1]))  #tensor([[ 1.],[ 1.],[ 1.],[ 1.]])

#获取维度 tensor.dim() , 还有tensor.min() tensor.max(), 转置tensor.t()(三个维度要写参数)   tensor.permute也是转置
t3 = torch.tensor(np.arange(24).reshape(2,3,4))
print(t3)
print(t3.size())
print(t3.transpose(0,1)) #2*3*4 变成了3*2*4

print(t3[0,1,2]) #第一块 第二行 第三列。 把6取出
print(t3[0,0:,2:]) #切片还是easy

