"""
# -*- coding: utf-8 -*-
# @Author : Sun JJ
# @File : load_data.py
# @Time : 2021/5/26 20:37
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

def load_data():
    """
    :return:training_affinity,training_pocket,training_protein,training_ligand,validation_affinity,validation_pocket,......
    """

    # training
    training_affinity = np.load('./data/Affinity/training_affinity.npy')
    training_pocket = np.load('./data/pocket/training_pocket.npy')
    training_protein = np.load('./data/protein/training_protein.npy')
    # training_complex = np.load('./data/entirety/training_complex.npy')
    training_ligand = np.load('./data/ligand/training_ligand.npy')
    training_ligandNet = np.load('./data/ligandNetwork/training_ligandNetwork.npy')
    training_ligandNet = training_ligandNet.reshape(training_ligandNet.shape[0],training_ligandNet.shape[1],
                                                    training_ligandNet.shape[2],1)

    # validation
    validation_affinity = np.load('./data/Affinity/validation_affinity.npy')
    validation_pocket = np.load('./data/pocket/validation_pocket.npy')
    validation_protein = np.load('./data/protein/validation_protein.npy')
    # validation_complex = np.load('./data/entirety/validation_complex.npy')
    validation_ligand = np.load('./data/ligand/validation_ligand.npy')
    validation_ligandNet = np.load('./data/ligandNetwork/validation_ligandNetwork.npy')
    validation_ligandNet = validation_ligandNet.reshape(validation_ligandNet.shape[0], validation_ligandNet.shape[1],
                                                        validation_ligandNet.shape[2], 1)

    return (training_affinity,training_pocket,training_protein,training_ligandNet,training_ligand,
            validation_affinity,validation_pocket,validation_protein,validation_ligandNet,validation_ligand)

def load_test_data():
    """
    :return: test_affinity,test_pocket,test_protein,test_ligand
    """

    # test
    test_affinity = np.load('./data/Affinity/test_affinity.npy')
    test_pocket = np.load('./data/pocket/test_pocket.npy')
    test_protein = np.load('./data/protein/test_protein.npy')
    # test_complex = np.load('./data/entirety/test_complex.npy')
    test_ligand = np.load('./data/ligand/test_ligand.npy')
    test_ligandNet = np.load('./data/ligandNetwork/test_ligandNetwork.npy')
    test_ligandNet = test_ligandNet.reshape(test_ligandNet.shape[0], test_ligandNet.shape[1],
                                            test_ligandNet.shape[2], 1)

    return test_affinity,test_pocket,test_protein,test_ligandNet,test_ligand




# (training_affinity,training_pocket,training_protein,training_ligandNet,training_ligand,
#  validation_affinity,validation_pocket,validation_protein,validation_ligandNet,validation_ligand) = load_data()
# test_affinity,test_pocket,test_protein,test_ligandNet,test_ligand = load_test_data()

# print('training_affinity:{}'.format(training_affinity.shape))
# print('training_pocket:{}'.format(training_pocket.shape))
# print('training_protein:{}'.format(training_protein.shape))
# print('training_ligandNet:{}'.format(training_ligandNet.shape))
# print('training_ligand:{}'.format(training_ligand.shape))
# print('====================================================')

# print('validation_affinity:{}'.format(validation_affinity.shape))
# print('validation_pocket:{}'.format(validation_pocket.shape))
# print('validation_protein:{}'.format(validation_protein.shape))
# print('validation_ligandNet:{}'.format(validation_ligandNet.shape))
# print('validation_ligand:{}'.format(validation_ligand.shape))
# print('====================================================')

# print('test_affinity:{}'.format(test_affinity.shape))
# print('test_pocket:{}'.format(test_pocket.shape))
# print('test_protein:{}'.format(test_protein.shape))
# print('test_ligandNet:{}'.format(test_ligandNet.shape))
# print('test_ligand:{}'.format(test_ligand.shape))
# print('====================================================')