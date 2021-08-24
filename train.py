"""
# -*- coding: utf-8 -*-
# @Author : Sun JJ
# @File : train.py
# @Time : 2021/5/26 22:10
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
import tensorflow as tf
import numpy as np
from metrics import cindex_score,c_index,RMSE,MAE,CORR,SD,get_cindex
from load_data import load_data,load_test_data
from keras import backend as K
from keras.callbacks import LearningRateScheduler

def PTA_Net(max_poc_len,max_pro_len,max_li_len,max_com_len,num_filters,filter_len1,filter_len2,filter_len3,filter_len4):
    # Input shape
    ligand_input = tf.keras.Input(shape=(max_li_len,), dtype = 'int32')
    poc_input = tf.keras.Input(shape = (max_poc_len,40), dtype = 'int32')
    pro_input = tf.keras.Input(shape = (max_pro_len,40),dtype = 'int32')
    # com_input = tf.keras.Input(shape = (max_com_len,),dtype = 'int32')
    ligandNet_input = tf.keras.Input(shape = (60,7,1))

    # CNN block of ligand representation
    li_p = tf.keras.layers.Embedding(input_dim = 3,output_dim = 128,input_length = 166)(ligand_input)
    print(li_p.shape)
    li_p = tf.keras.layers.Conv1D(filters = num_filters,kernel_size = filter_len1,
                                  padding = 'same',activation = 'relu')(li_p)
    print(li_p.shape)
    li_p = tf.keras.layers.BatchNormalization()(li_p)
    li_p = tf.keras.layers.Conv1D(filters = num_filters * 2,kernel_size = filter_len1,
                                  padding = 'same',activation = 'relu')(li_p)
    print(li_p.shape)
    li_p = tf.keras.layers.BatchNormalization()(li_p)
    li_p = tf.keras.layers.Conv1D(filters = num_filters * 3,kernel_size = filter_len1,
                                  padding = 'same',activation = 'relu')(li_p)
    print(li_p.shape)
    li_p = tf.keras.layers.BatchNormalization()(li_p)
    li_p = tf.keras.layers.GlobalAveragePooling1D(data_format='channels_first')(li_p)
    # li_p = tf.keras.layers.GlobalMaxPool1D(data_format='channels_first')(li_p)
    li_p = tf.keras.layers.Dropout(0.3)(li_p)
    print(li_p.shape)

    # CNN block of ligandNet representation
    liNet_p = tf.keras.layers.Conv2D(filters = num_filters,kernel_size = filter_len2,
                                     padding = 'same',activation = 'relu',data_format = 'channels_last')(ligandNet_input)
    liNet_p = tf.keras.layers.BatchNormalization()(liNet_p)
    liNet_p = tf.keras.layers.Conv2D(filters=num_filters * 2, kernel_size=filter_len2,
                                     padding='same', activation='relu', data_format='channels_last')(liNet_p)
    liNet_p = tf.keras.layers.BatchNormalization()(liNet_p)
    liNet_p = tf.keras.layers.Conv2D(filters=num_filters * 3, kernel_size=filter_len2,
                                     padding='same', activation='relu', data_format='channels_last')(liNet_p)
    liNet_p = tf.keras.layers.BatchNormalization()(liNet_p)
    liNet_p = tf.keras.layers.GlobalAveragePooling2D(data_format = 'channels_last')(liNet_p)
    liNet_p = tf.keras.layers.Dropout(0.3)(liNet_p)


    # CNN block of pocket representation
    poc_p = tf.keras.layers.Embedding(input_dim = 3,output_dim = 128,input_length = max_poc_len)(poc_input)
    print(poc_p.shape)
    poc_p = tf.keras.layers.Conv2D(filters = num_filters,kernel_size = filter_len2,
                                   padding = 'same',activation = 'relu',data_format='channels_last')(poc_p)
    print(poc_p.shape)
    poc_p = tf.keras.layers.BatchNormalization()(poc_p)
    poc_p = tf.keras.layers.Conv2D(filters = num_filters * 2,kernel_size = filter_len2,
                                   padding = 'same',activation = 'relu',data_format='channels_last')(poc_p)
    print(poc_p.shape)
    poc_p = tf.keras.layers.BatchNormalization()(poc_p)
    poc_p = tf.keras.layers.Conv2D(filters = num_filters * 3,kernel_size = filter_len2,
                                   padding = 'same',activation = 'relu',data_format='channels_last')(poc_p)
    print(poc_p.shape)
    poc_p = tf.keras.layers.BatchNormalization()(poc_p)
    poc_p = tf.keras.layers.GlobalAveragePooling2D(data_format='channels_last')(poc_p)
    # poc_p = tf.keras.layers.GlobalMaxPool1D(data_format='channels_first')(poc_p)
    poc_p = tf.keras.layers.Dropout(0.3)(poc_p)
    print(poc_p.shape)


    # CNN block of protein representation
    pro_p = tf.keras.layers.Embedding(input_dim=3, output_dim=128, input_length=max_pro_len)(pro_input)
    print(pro_p.shape)
    pro_p = tf.keras.layers.Conv2D(filters = num_filters,kernel_size = filter_len3,
                                   padding = 'same',activation = 'relu',data_format='channels_last')(pro_p)
    print(pro_p.shape)
    pro_p = tf.keras.layers.BatchNormalization()(pro_p)
    pro_p = tf.keras.layers.Conv2D(filters = num_filters * 2,kernel_size = filter_len3,
                                   padding = 'same',activation = 'relu',data_format='channels_last')(pro_p)
    print(pro_p.shape)
    pro_p = tf.keras.layers.BatchNormalization()(pro_p)
    pro_p = tf.keras.layers.Conv2D(filters = num_filters * 3,kernel_size = filter_len3,
                                   padding = 'same',activation = 'relu',data_format='channels_last')(pro_p)
    print(pro_p.shape)
    pro_p = tf.keras.layers.BatchNormalization()(pro_p)
    pro_p = tf.keras.layers.GlobalAveragePooling2D(data_format='channels_last')(pro_p)
    # pro_p = tf.keras.layers.GlobalMaxPool1D(data_format='channels_first')(pro_p)
    pro_p = tf.keras.layers.Dropout(0.3)(pro_p)
    print(pro_p.shape)

    # print(li_p.shape,poc_p.shape,pro_p.shape,liNet_p.shape)

    # Combined representation
    c_p = tf.keras.layers.concatenate([li_p,liNet_p,poc_p,pro_p], axis=-1, name='concatenation')

    # three layers of full connection
    FC = tf.keras.layers.Dense(1024, activation='relu')(c_p)
    FC = tf.keras.layers.Dropout(0.2)(FC)
    FC = tf.keras.layers.Dense(1024, activation='relu')(FC)
    FC = tf.keras.layers.Dropout(0.2)(FC)
    FC = tf.keras.layers.Dense(512, activation='relu')(FC)

    y = tf.keras.layers.Dense(1, kernel_initializer='normal')(FC)

    model = tf.keras.Model(inputs=[ligand_input,poc_input,pro_input,ligandNet_input], outputs=[y])
    model.compile(optimizer='adam', loss='mae', metrics=[cindex_score])   # mean_squared_error

    print(model.summary())
    return model


def scheduler(epoch):
    # 每隔20个epoch，学习率减小为原来的1/2
    if epoch % 20 == 0 and epoch != 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.5)
        print("lr changed to {}".format(lr * 0.5))
    return K.get_value(model.optimizer.lr)
reduce_lr = LearningRateScheduler(scheduler)

max_poc_len = 63
max_pro_len = 1000
max_li_len = 166
max_com_len = 1100
num_filters = 16
filter_len1 = 5
filter_len2 = 3
filter_len3 = 5
filter_len4 = 3


# load data

(training_affinity,training_pocket,training_protein,training_ligandNet,training_ligand,
 validation_affinity,validation_pocket,validation_protein,validation_ligandNet,validation_ligand) = load_data()
test_affinity,test_pocket,test_protein,test_ligandNet,test_ligand = load_test_data()

# load model

model = PTA_Net(max_poc_len,max_pro_len,max_li_len,max_com_len,num_filters,filter_len1,filter_len2,filter_len3,filter_len4)

checkpoint_save_path = "checkpoint/drop0.2lr_B/PTA.ckpt"
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)
es = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss',patience = 10,min_delta = 0.001,mode = 'min')

history = model.fit(x = ([training_ligand,training_pocket,training_protein,training_ligandNet]),y = training_affinity,
                    batch_size = 64,epochs = 100,
                    validation_data = ([validation_ligand,validation_pocket,validation_protein,validation_ligandNet],validation_affinity),
                    shuffle = False,callbacks = [cp_callback,reduce_lr])




# train_predicted_labels = model.predict([training_ligand,training_pocket,training_protein,training_ligandNet])
# train_predicted_labels_list = train_predicted_labels.tolist()
# train_predicted_labels_list = [i[0] for i in train_predicted_labels_list]
# rmse = RMSE(training_affinity,train_predicted_labels)
# mae = MAE(training_affinity,train_predicted_labels)
# corr = CORR(training_affinity,train_predicted_labels_list)
# sd = SD(training_affinity,train_predicted_labels)
# ci = get_cindex(training_affinity,train_predicted_labels)
# print(rmse,mae,corr,sd,ci)
# print('=============================')
# validation_predicted_labels = model.predict([validation_ligand,validation_pocket,validation_protein,validation_ligandNet])
# validation_predicted_labels_list = validation_predicted_labels.tolist()
# validation_predicted_labels_list = [i[0] for i in validation_predicted_labels_list]
# rmse = RMSE(validation_affinity,validation_predicted_labels)
# mae = MAE(validation_affinity,validation_predicted_labels)
# corr = CORR(validation_affinity,validation_predicted_labels_list)
# sd = SD(validation_affinity,validation_predicted_labels)
# ci = get_cindex(validation_affinity,validation_predicted_labels)
# print(rmse,mae,corr,sd,ci)
# print('=============================')
# test_predicted_labels = model.predict([test_ligand,test_pocket,test_protein,test_ligandNet])
# test_predicted_labels_list = test_predicted_labels.tolist()
# test_predicted_labels_list = [i[0] for i in test_predicted_labels_list]
# rmse = RMSE(test_affinity,test_predicted_labels)
# mae = MAE(test_affinity,test_predicted_labels)
# corr = CORR(test_affinity,test_predicted_labels_list)
# sd = SD(test_affinity,test_predicted_labels)
# ci = get_cindex(test_affinity,test_predicted_labels)
# print(rmse,mae,corr,sd,ci)