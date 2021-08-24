"""
# -*- coding: utf-8 -*-
# @Author : Sun JJ
# @File : protein_representation_prepare.py
# @Time : 2021/5/26 14:47
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
MAX_SEQ_LEN = 1000


# test

count1,count2 = 0,0
test_protein_dict = {}
test_protein_list = []

for home, dirs, files in os.walk(r'D:\PTA4\data\test\global'):
    for file in files:
        protein_data_np = np.zeros((1000,40),dtype = np.float16)
        pdbid = file[:4]
        protein_data_pd = pd.read_csv('./data/test/global/' + file)
        if len(protein_data_pd) <= MAX_SEQ_LEN:
            for i in range(len(protein_data_pd)):
                protein_data_np[i] = np.array(protein_data_pd.iloc[0][1:-1])
        else:
            for i in range(MAX_SEQ_LEN):
                protein_data_np[i] = np.array(protein_data_pd.iloc[0][1:-1])
        test_protein_dict[pdbid] = protein_data_np
        count1 += 1
        print('count1:{}'.format(count1))
for i in test_pdbid:
    test_protein_list.append(test_protein_dict[i])
    count2 += 1
    print('count2:{}'.format(count2))
np.save('./data/protein/test_protein.npy', np.array(test_protein_list))


# training

count1,count2 = 0,0
training_protein_dict = {}
training_protein_list = []

for home, dirs, files in os.walk(r'D:\PTA4\data\training\global'):
    for file in files:
        protein_data_np = np.zeros((1000,40),dtype = np.float16)
        pdbid = file[:4]
        protein_data_pd = pd.read_csv('./data/training/global/' + file)
        if len(protein_data_pd) <= MAX_SEQ_LEN:
            for i in range(len(protein_data_pd)):
                protein_data_np[i] = np.array(protein_data_pd.iloc[0][1:-1])
        else:
            for i in range(MAX_SEQ_LEN):
                protein_data_np[i] = np.array(protein_data_pd.iloc[0][1:-1])
        training_protein_dict[pdbid] = protein_data_np
        count1 += 1
        print('count1:{}'.format(count1))
for i in training_pdbid:
    training_protein_list.append(training_protein_dict[i])
    count2 += 1
    print('count2:{}'.format(count2))
np.save('./data/protein/training_protein.npy', np.array(training_protein_list))


# validation

count1,count2 = 0,0
validation_protein_dict = {}
validation_protein_list = []

for home, dirs, files in os.walk(r'D:\PTA4\data\validation\global'):
    for file in files:
        protein_data_np = np.zeros((1000,40),dtype = np.float16)
        pdbid = file[:4]
        protein_data_pd = pd.read_csv('./data/validation/global/' + file)
        if len(protein_data_pd) <= MAX_SEQ_LEN:
            for i in range(len(protein_data_pd)):
                protein_data_np[i] = np.array(protein_data_pd.iloc[0][1:-1])
        else:
            for i in range(MAX_SEQ_LEN):
                protein_data_np[i] = np.array(protein_data_pd.iloc[0][1:-1])
        validation_protein_dict[pdbid] = protein_data_np
        count1 += 1
        print('count1:{}'.format(count1))
for i in validation_pdbid:
    validation_protein_list.append(validation_protein_dict[i])
    count2 += 1
    print('count2:{}'.format(count2))
np.save('./data/protein/validation_protein.npy', np.array(validation_protein_list))
