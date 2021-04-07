# -*- coding = utf-8 -*-
# @Time : 2021/4/3 10:09 下午
# @Author : Pan
# @Software: PyCharm


#torchvision.datasets  比如 torchvision.datasets.MNIST
#torchtext.datasets  比如 torchtext.datasets.IMDB
#transforms.ToTensor()  将图片对象转换成tensor
#transfroms.Normalize图片规范化， 每个通道的均值和标准差。形状必须要和通道数相同
#transfroms.compose （前面两个的结合）传入列表。 [transforms.ToTensor(),transfroms.Normalize]

from torchvision import transforms
from torchvision.datasets import MNIST


mnist = MNIST(root="./data",train=True,download=False)
print(mnist)  #Dataset
# print(mnist[0])   #(<PIL.Image.Image image mode=L size=28x28 at 0x7FFC332611C0>, tensor(5))
# print(mnist[0][0].show())  #跳出一张5的图片

ret = transforms.ToTensor()(mnist[0][0])  #可以直接传(mnist[0][0])，因为有call方法
print(ret.size())   #torch.Size([1, 28, 28])
print(ret[0])

