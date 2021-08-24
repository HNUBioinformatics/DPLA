"""
# -*- coding: utf-8 -*-
# @Author : Sun JJ
# @File : ligand_representation.py
# @Time : 2021/5/25 20:55
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
from rdkit import Chem
from rdkit.Chem import MACCSkeys

f = open("./data/still_miss_mol_pdbid.txt","r",encoding='ISO-8859-15')   #设置文件对象
still_miss_mol_pdbid = f.read().split('\n')     #将txt文件的所有内容读入到字符串str中
f.close()   #将文件关闭



datasets = ['training','validation','test']

count = 0

for dataset in datasets:
    data_pd = pd.read_csv('./data/' + dataset + '_smi.csv')
    all_features = []
    count1 = 0

    for i in range(len(data_pd)):
        pdbid = data_pd.iloc[i]['pdbid']
        smile = data_pd.iloc[i]['smiles']
        features = []

        if pdbid not in still_miss_mol_pdbid:
            mol = Chem.MolFromSmiles(smile)
            # mol == None
            if mol == None:
                mol = Chem.MolFromMolFile('./data/Miss_Mol/' + pdbid + '_ligand.mol')

            fingerprints = MACCSkeys.GenMACCSKeys(mol)
            for i in range(1, len(fingerprints.ToBitString())):
                features.append(float(fingerprints.ToBitString()[i]))

            all_features.append(features)
            count += 1
        print(count)

    all_features_np = np.array(all_features)
    file_path = './data/ligand/' + dataset + '_ligand.npy'
    np.save(file_path, all_features_np)