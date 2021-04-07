# -*- coding = utf-8 -*-
# @Time : 2021/3/28 3:02 下午
# @Author : Pan
# @Software: PyCharm

import torch
import numpy as np

t1 = torch.ones([3,4],dtype=torch.double)  #指定dtype创建
print(t1)

t2 = torch.tensor(np.arange(12).reshape(3,4),dtype=torch.float32)
print(t2)

print(t1.int())  #直接修改数据类型


t3 = t2.new_ones([2,2])  #与ones稍有区别
print(t3)

print(t1+t2)
print(torch.add(t1,t2))  #这两个还是一样的。

t1.add_(t2) #原地修改。t1变成了t1+t2的值  很多方法都可以带下划线。（就不用取再用变量取接收了）

print("______________________________________________")


print(torch.cuda.is_available())  #False.不能用gpu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #有了这个三元判别式就不管有没有cuda了
print(torch.zeros([2,3],device=device))  #基于device创建
print(t1.to(device)) #有cuda就会把t1这个tensor 扔到cuda上去


