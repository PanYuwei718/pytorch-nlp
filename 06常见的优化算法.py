# -*- coding = utf-8 -*-
# @Time : 2021/4/2 10:53 下午
# @Author : Pan
# @Software: PyCharm

#随机梯度下降：随机从样本中抽出一个样本进行梯度更新
#小批量梯度下降 找一波数据计算梯度，使用均值更新参数
#动量法，基于梯度的指数加权平均。 （前面的梯度乘一个参数加上当前的梯度乘一个参数，作为新的梯度）相当于给梯度赋予之前梯度的权重。）
#还有RMSProp对梯度进行平方加权

#adagrad自适应学习率（稀疏数据下表现较好）
#adam 是momentum 和 RMSProp的结合

import MeCab
mecab = MeCab.Tagger("-Ochasen")
sentence = '太郎はこの本を二郎を見た女性に渡した。'
print(mecab.parse(sentence))
