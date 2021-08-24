"""
# -*- coding: utf-8 -*-
# @Author : Sun JJ
# @File : pocket_representation_prepare.py
# @Time : 2021/5/26 10:28
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

import os
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

f = open("./data/still_miss_mol_pdbid.txt","r",encoding='ISO-8859-15')   #设置文件对象
still_miss_mol_pdbid = f.read().split('\n')     #将txt文件的所有内容读入到字符串str中
f.close()   #将文件关闭

# 一些常量
MAX_SEQ_LEN = 63

# test

count1 = 0
count2 = 0
test_pocket_dict = {}
test_pocket_list = []

for home, dirs, files in os.walk(r'D:\PTA4\data\test\pocket'):
    for file in files:
        pocket_data_np = np.zeros((63,40),dtype = np.float16)
        pdbid = file[:4]
        pocket_data_pd = pd.read_csv('./data/test/pocket/' + file)
        if len(pocket_data_pd) <= MAX_SEQ_LEN:
            for i in range(len(pocket_data_pd)):
                pocket_data_np[i] = np.array(pocket_data_pd.iloc[i][2:])
        else:
            for i in range(MAX_SEQ_LEN):
                pocket_data_np[i] = np.array(pocket_data_pd.iloc[i][2:])
        test_pocket_dict[pdbid] = pocket_data_np
        count1 += 1
        print('count1:{}'.format(count1))
print('test_pocket_dict:{}'.format(len(test_pocket_dict)))

for i in test_pdbid:
    test_pocket_list.append(test_pocket_dict[i])
    count2 += 1
    print('count2:{}'.format(count2))
print('test_pocket_list:{}'.format(len(test_pocket_list)))
np.save('./data/pocket/test_pocket.npy', np.array(test_pocket_list))


# training

count1 = 0
count2 = 0
training_pocket_dict = {}
training_pocket_list = []

for home, dirs, files in os.walk(r'D:\PTA4\data\training\pocket'):
    for file in files:
        pocket_data_np = np.zeros((63,40),dtype = np.float16)
        pdbid = file[:4]
        pocket_data_pd = pd.read_csv('./data/training/pocket/' + file)
        if len(pocket_data_pd) <= MAX_SEQ_LEN:
            for i in range(len(pocket_data_pd)):
                pocket_data_np[i] = np.array(pocket_data_pd.iloc[i][2:])
        else:
            for i in range(MAX_SEQ_LEN):
                pocket_data_np[i] = np.array(pocket_data_pd.iloc[i][2:])
        training_pocket_dict[pdbid] = pocket_data_np
        count1 += 1
        print('count1:{}'.format(count1))
print('training_pocket_dict:{}'.format(len(training_pocket_dict)))

for i in training_pdbid:
    training_pocket_list.append(training_pocket_dict[i])
    count2 += 1
    print('count2:{}'.format(count2))
print('training_pocket_list:{}'.format(len(training_pocket_list)))
np.save('./data/pocket/training_pocket.npy', np.array(training_pocket_list))


# validation

count1 = 0
count2 = 0
validation_pocket_dict = {}
validation_pocket_list = []

for home, dirs, files in os.walk(r'D:\PTA4\data\validation\pocket'):
    for file in files:
        pocket_data_np = np.zeros((63,40),dtype = np.float16)
        pdbid = file[:4]
        pocket_data_pd = pd.read_csv('./data/validation/pocket/' + file)
        if len(pocket_data_pd) <= MAX_SEQ_LEN:
            for i in range(len(pocket_data_pd)):
                pocket_data_np[i] = np.array(pocket_data_pd.iloc[i][2:])
        else:
            for i in range(MAX_SEQ_LEN):
                pocket_data_np[i] = np.array(pocket_data_pd.iloc[i][2:])
        validation_pocket_dict[pdbid] = pocket_data_np
        count1 += 1
        print('count1:{}'.format(count1))
print('validation_pocket_dict:{}'.format(len(validation_pocket_dict)))

for i in validation_pdbid:
    validation_pocket_list.append(validation_pocket_dict[i])
    count2 += 1
    print('count2:{}'.format(count2))
print('validation_pocket_list:{}'.format(len(validation_pocket_list)))
np.save('./data/pocket/validation_pocket.npy', np.array(validation_pocket_list))