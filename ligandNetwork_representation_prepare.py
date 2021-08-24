"""
# -*- coding: utf-8 -*-
# @Author : Sun JJ
# @File : ligandNetwork_representation_prepare.py
# @Time : 2021/8/10 11:19
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
import networkx as nx
from rdkit import Chem


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


datasets = ['training','validation','test']

count = 0

for dataset in datasets:
    data_pd = pd.read_csv('./data/' + dataset + '_smi.csv')
    all_features = []

    for i in range(len(data_pd)):
        pdbid = data_pd.iloc[i]['pdbid']
        smile = data_pd.iloc[i]['smiles']

        if pdbid not in still_miss_mol_pdbid:
            mol = Chem.MolFromSmiles(smile)
            # mol == None
            if mol == None:
                mol = Chem.MolFromMolFile('./data/Miss_Mol/' + pdbid + '_ligand.mol')

            c_size = mol.GetNumAtoms()
            edges = []
            for bond in mol.GetBonds():
                edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
            g = nx.Graph(edges).to_directed()

            a = nx.average_neighbor_degree(g)
            try:
                b = nx.eccentricity(g)
            except:
                b = {}
                for j in range(c_size):
                    b[j] = 0
            c = nx.clustering(g)
            d = nx.degree(g)
            e = nx.degree_centrality(g)
            f = nx.betweenness_centrality(g)
            h = nx.closeness_centrality(g)

            net_features = np.zeros([60,7])

            if len(a) > 60:
                for k in range(60):
                    net_features[k][0] = a[k]
                    net_features[k][1] = b[k]
                    net_features[k][2] = c[k]
                    net_features[k][3] = d[k]
                    net_features[k][4] = e[k]
                    net_features[k][5] = f[k]
                    net_features[k][6] = h[k]
            else:
                if count == 4111:
                    for k in range(len(a)):
                        if k == 10 or k == 11:
                            net_features[k][0] = a[k + 1]
                            net_features[k][1] = b[k + 1]
                            net_features[k][2] = c[k + 1]
                            net_features[k][3] = d[k + 1]
                            net_features[k][4] = e[k + 1]
                            net_features[k][5] = f[k + 1]
                            net_features[k][6] = h[k + 1]
                        else:
                            net_features[k][0] = a[k]
                            net_features[k][1] = b[k]
                            net_features[k][2] = c[k]
                            net_features[k][3] = d[k]
                            net_features[k][4] = e[k]
                            net_features[k][5] = f[k]
                            net_features[k][6] = h[k]
                else:
                    for k in range(len(a)):
                        net_features[k][0] = a[k]
                        net_features[k][1] = b[k]
                        net_features[k][2] = c[k]
                        net_features[k][3] = d[k]
                        net_features[k][4] = e[k]
                        net_features[k][5] = f[k]
                        net_features[k][6] = h[k]

            all_features.append(net_features)

            count += 1
            print(count)
    all_features_np = np.array(all_features)
    file_name = './data/ligandNetwork/' + dataset + '_ligandNetwork.npy'
    np.save(file_name,all_features_np)

