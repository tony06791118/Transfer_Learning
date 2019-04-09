#　程式碼開頭加這串保平安
#
#                   ooOoo
#                  o8888888o
#                  88" . "88
#                  (| -_- |)
#                  O\  =  /O
#               ___/`---//\___
#             .//  \\|     |//  `.
#            /  \\|||  :  |||//  \
#           /  _||||| -:- |||||-  \
#           |   | \\\  -  /// |   |
#           | \_|  ////\---/////  |   |
#           \  .-\_  `-`  __/-. /
#         __`. .//  /--.--\  `. . _
#      ."" //<  `.___\_<|>_/___.//  >//"".
#     | | :  `- `.;`\ _ /`;.`/ - ` : | |
#     \  \ `-.   \ _\ /_ /   .-` /  /
#======`-.____`-.___\_____/___.-`____.-//======
#                   `=---=//
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#　　　　佛祖保佑　bug退散　永不當機
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 16:13:21 2019

@author: sai-pc

測試transfer learning：一個人家已經train好的ResNet50模型，拿來修改最後兩層做transfer learning


"""

from keras.applications import ResNet50
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Flatten, GlobalAveragePooling2D
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
import os
import sys
import time

#手動拆層建立模型：
def create_manual_model(file_path):
    #建立一個沒有給weight的Rest50 Model
    ResNet_model_no_top_layer = ResNet50(include_top=False, pooling='avg', weights = None)
    ##print("ResNet_model_no_top_layer layers:")
    ##print(len(ResNet_model_no_top_layer.layers))
    
    #取出input層 和 output層
    m_input = ResNet_model_no_top_layer.input
    m_output = ResNet_model_no_top_layer.output
    
    #建立final model
    final_Model = Model(inputs = m_input, outputs = m_output)
    ##print("final_Model layers:")
    ##print(len(final_Model.layers))
    
    #欲拆解的完整模型
    #resnet_weights_path = '/Users/sai-pc/Desktop/另外實作/transfer_learning/Code/utils/dl \
    #                       /ResNet50/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
    resnet_weights_path = file_path + '/utils/dl/ResNet50/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
    ResNet_model_have_top_layer = ResNet50(include_top=True, pooling='avg', weights = resnet_weights_path)
    
    ##print("ResNet_model_have_top_layer layers:")
    ##print(len(ResNet_model_have_top_layer.layers))
    
    for i, layer in enumerate(final_Model.layers):
        print(i, layer.name)
        layer.set_weights(ResNet_model_have_top_layer.layers[i].get_weights())
    
    #----------組裝模型(functional_api)----------
    #你的預測的類別有幾種,num_classes就填多少
    num_classes = 2
    #最後一層, pred層
    pred_layer = Dense(num_classes, activation='softmax')(final_Model.output)
    
    my_new_manual_model = Model(inputs = final_Model.input, outputs = pred_layer)
    
    #前面模型不訓練(my_new_model的第一層是拔掉最後一層的ResNet),只訓練後面(新增的softmax那層)
    for i in range(0, len(my_new_manual_model.layers) -1):
        my_new_manual_model.layers[i].trainable = False
    
    #編譯模型
    # We are calling the compile command for some python object. 
    # Which python object is being compiled? Fill in the answer so the compile command works.
    #optimizer = 模型優化法, 'sgd' = 隨機梯度下降法, loss function = 損失函數
    my_new_manual_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return my_new_manual_model

#建立RestNet50模型, 並用他設計好的weight：
def create_model(file_path):
    # num_classes is the number of categories your model chooses between for each prediction#
    #你的預測的類別有幾種,num_classes就填多少
    num_classes = 2
    #model位置
    #resnet_weights_path = '/Users/sai-pc/Desktop/另外實作/transfer_learning/Code/utils/dl/ResNet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
    resnet_weights_path = file_path + '/utils/dl/ResNet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'    
    my_new_model = Sequential()
    #其中include_top = False ->把ResNet50這模型最後一層拔掉
    my_new_model.add(ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path))
    #最後一層再加上一層sofmax的層
    my_new_model.add(Dense(num_classes, activation='softmax'))
    
    # The value below is either True or False.  If you choose the wrong answer, your modeling results
    # won't be very good.  Recall whether the first layer should be trained/changed or not.
    #前面模型不訓練(my_new_model的第一層是拔掉最後一層的ResNet),只訓練後面(新增的softmax那層)
    my_new_model.layers[0].trainable = False  
    
    #編譯模型
    # We are calling the compile command for some python object. 
    # Which python object is being compiled? Fill in the answer so the compile command works.
    #optimizer = 模型優化法, 'sgd' = 隨機梯度下降法, loss function = 損失函數
    my_new_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    
    #編譯完回傳
    return my_new_model
    
#train model use: Data Augmentation
def train_mondel(my_new_model, file_path):
    #圖片image size 為 224*224
    image_size = 224
    
    #Data Augmentation 簡單來說，就是想辦法從舊照片生出新照片，讓總訓練的資料數增加。
    #我們可以把照片旋轉、切割、放大縮小、改變顏色等等做各種變形。
    #ImageDataGenerator() 為一種圖片生成器
    #有做data aug的data生成器
    data_generator_with_aug = ImageDataGenerator(
                                             horizontal_flip = True,
                                             width_shift_range = 0.2,
                                             height_shift_range = 0.2)
    #沒有做data aug的data產生器
    data_generator_no_aug = ImageDataGenerator()
    
    #有做data aug的當training data
    train_generator = data_generator_with_aug.flow_from_directory(
            #directory = '/Users/sai-pc/Desktop/另外實作/transfer_learning/Code/train',
            directory = file_path + '/train',
            target_size=(image_size, image_size),
            batch_size=10,
            class_mode='categorical')

    validation_generator = data_generator_no_aug.flow_from_directory(
            #directory = '/Users/sai-pc/Desktop/另外實作/transfer_learning/Code/val',
            directory = file_path + '/val',
            target_size=(image_size, image_size),
            class_mode='categorical')

    my_new_model.fit_generator(
            train_generator,
            epochs = 3,
            steps_per_epoch = 19,
            validation_data=validation_generator)
    
def main():
    
    #執行起始時間
    start_time = time.time()
    
    
    #file_path = 這個python檔案的所在絕對路徑
    file_path = os.path.abspath('')

    #create my model
    #my_new_model = create_model(file_path)
    ##print("----------")
    ##print("my_new_model.summary()")
    ##my_new_model.summary()
    ##print("----------")
    
    #create my manual model
    my_new_manual_model = create_manual_model(file_path)
    ##print("----------")
    ##print("my_new_manual_model.summary()")
    ##my_new_manual_model.summary()
    ##print("----------")
    
    #train model
    #train_mondel(my_new_model)
    train_mondel(my_new_manual_model, file_path)   
    
    #執行結束時間
    end_time = time.time()
    
    #計算程式執行時間
    run_time = end_time - start_time
    print('all run time : ' + str(run_time))
    
if __name__ == '__main__':
    main()