"""
# -*- coding: utf-8 -*-
# @Author : Sun JJ
# @File : affinity_prepare.py
# @Time : 2021/5/31 9:58
# code is far away from bugs with the god animal protecting
#         ┌─┐       ┌─┐
#      ┌──┘ ┴───────┘ ┴──┐
#      │                 │
#      │       ───       │
#      │  ─┬┘       └┬─  │
#      │                 │
#      │       ─┴─       │
#      │                 │
#      └───┐         ┌───┘
#          │         │
#          │         │
#          │         │
#          │         └──────────────┐
#          │                        │
#          │                        ├─┐
#          │                        ┌─┘
#          │                        │
#          └─┐  ┐  ┌───────┬──┐  ┌──┘
#            │ ─┤ ─┤       │ ─┤ ─┤
#            └──┴──┘       └──┴──┘
"""


import numpy as np
import pandas as pd



f = open("./data/training_pdbid.txt","r",encoding='ISO-8859-15')   #设置文件对象
training_pdbid = f.read().split('\n')     #将txt文件的所有内容读入到字符串str中
f.close()   #将文件关闭

f = open("./data/validation_pdbid.txt","r",encoding='ISO-8859-15')   #设置文件对象
validation_pdbid = f.read().split('\n')     #将txt文件的所有内容读入到字符串str中
f.close()   #将文件关闭

f = open("./data/test_pdbid.txt","r",encoding='ISO-8859-15')   #设置文件对象
test_pdbid = f.read().split('\n')     #将txt文件的所有内容读入到字符串str中
f.close()   #将文件关闭

all_affinity_dict = {}
all_affinity = pd.read_csv('./data/Affinity/affinity_data.csv')

for i in range(len(all_affinity)):
    pdbid = all_affinity.iloc[i]['pdbid']
    affinity = all_affinity.iloc[i]['-logKd/Ki']
    all_affinity_dict[pdbid] = affinity

# test
affinity_list = []
count = 0

for i in test_pdbid:
    affinity_list.append(all_affinity_dict[i])
    count += 1
    print(count)

affinity_np = np.array(affinity_list)
np.save('./data/Affinity/test_affinity.npy',affinity_np)



# validation
affinity_list = []
count = 0

for i in validation_pdbid:
    affinity_list.append(all_affinity_dict[i])
    count += 1
    print(count)

affinity_np = np.array(affinity_list)
np.save('./data/Affinity/validation_affinity.npy',affinity_np)



# training
affinity_list = []
count = 0

for i in training_pdbid:
    affinity_list.append(all_affinity_dict[i])
    count += 1
    print(count)

affinity_np = np.array(affinity_list)
np.save('./data/Affinity/training_affinity.npy',affinity_np)