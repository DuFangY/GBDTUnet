# -*- coding: UTF-8 -*-
# @Time : 2022/6/24 20:37
# @File : four_csv_two.py
# @Sofrware : PyCharm
# @Author : Du Fangyuan
import numpy as np
import pandas as pd

def label_1(): #其他为0 1还是1
    data = pd.read_csv("../slice_feature_all.csv")
    # data = pd.read_csv("../test.csv")
    data = data.fillna(0)
    y = data.zLabel.values
    X = data.drop("zLabel", axis=1)
    y[y != 1] = 0
    X.insert(0,'zLabel',y)
    X.to_csv('./1.csv',index=None)
    # X.to_csv('./1_test.csv',index=None)
    print("标签1分类csv转化完成")

label_1()