# -*- coding = utf-8 -*-
# @Time : 2021/4/3 8:56 下午
# @Author : Pan
# @Software: PyCharm


#数据集类 torch.Dataset  __getitem__ 通过传入索引的方式获取数据（dataset[i]） , __len__ 获取数据个数
#DataLoader类，dataset参数（传入实例化后的my_dataset对象）。batch_size参数。 shuffle=True  drop_last=True扔掉最后多余的数据
#SMS Spam数据集。

import  torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

data_path = r"./SMSSpamCollection"  #注意 反斜杠是/，不是\

#完成数据集类
class  MyDataset(Dataset):
    def __init__(self):
        self.lines = open(data_path).readlines()

    def __getitem__(self, index):   #获取单条数据，分成标签+数据
        cur_line = self.lines[index].strip()  #strip() 是去除字符串首位的字符。里面可以加'a'表示去除首位的a
        label = cur_line[:4].strip()
        content = cur_line[4:].strip()
        return label,content

    def __len__(self):           #返回数据总数量
        return len(self.lines)



my_dataset = MyDataset()
data_loader = DataLoader(dataset=my_dataset,batch_size=2,shuffle=True,drop_last=True) #batchsize = 2,所以是这种格式[('spam', 'ham'), ('Todays Voda numbers ending 5226 are selected to receive a ?350 award. If you hava a match please call 08712300220 quoting claim code 1131 standard rates app', 'True. It is passable. And if you get a high score and apply for phd, you get 5years of salary. So it makes life easier.')]

if __name__ == '__main__':
    print(my_dataset)   #<__main__.MyDataset object at 0x7fe600261280>
    print(my_dataset[0])  #('ham', 'Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...')
    for index,(label,context) in enumerate(data_loader):  #data_loader[0] 不行
        print(index,label,context)
        print("*"*100)



