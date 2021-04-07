# -*- coding = utf-8 -*-
# @Time : 2021/4/6 10:37 午後
# @Author : Pan
# @Software: PyCharm

#torch.nn.Embedding(词典数量，embedding维度)
#batch=5 每个句子有十个单词。embedding（20，4），最后每个句子的维度是[5,10,4]

import torch
import re
import os
from torch.utils.data import DataLoader,Dataset


#完成数据集的准备
class ImdbDataset(Dataset):
    def __init__(self,train=True):
        self.train_data_path = r'./aclImdb/train'
        self.test_data_path = r'./aclImdb/test'
        data_path = self.train_data_path if train else self.test_data_path

        #把所有的文件名放入列表
        temp_data_path = [os.path.join(data_path+'/pos'),os.path.join(data_path+'/neg')]
        self.total_file_path = []
        for pos_neg_path in temp_data_path:
            file_name_list = os.listdir(pos_neg_path)  #os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表。
            file_name_list = [os.path.join(pos_neg_path,i)for i in file_name_list if i.endswith(".txt")]
            self.total_file_path.extend(file_name_list)



    def __getitem__(self, item):
        file_path = self.total_file_path[item]

       #获取label
        label_str = file_path.split("\\") #这里不能直接加r，有问题。。


    def __len__(self):
        pass

if __name__ == '__main__':
    imdb_dataset = ImdbDataset()
    print(imdb_dataset[0])

