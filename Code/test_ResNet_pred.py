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

測試模型：一個人家已經train好的ResNet50模型，拿來預測圖片是否為熱狗。
"""

#step 1:
#import一些kaggle寫好「把照片讀進來(load_my_image)、對照片做 convolution(apply_conv_to_image)、顯示照片(show)」等功能
import sys
sys.path.append('/Users/sai-pc/Desktop/另外實作/transfer_learning/Code/utils/dl')
from exercise_1 import load_my_image, apply_conv_to_image, show
from os.path import join
import numpy as np
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from decode_predictions import decode_predictions
from IPython.display import Image, display

#step2:
#Kaggle 用水平線 convolution 為範例，上面 +1 下面 -1 代表被加權後，圖片上下顏色一樣的地方會變成零，
#上下顏色不一樣的地方就不是零，於是水平特徵被凸顯出來了。
#垂直線 convolution 依此類推，應該設計成左邊 +1 右邊 -1，讓左右不一樣的地方凸顯出來。
def design_convolution():
    # Detects bright pixels over dark pixels. (水平特徵)
    horizontal_line_conv = [[1, 1], 
                            [-1, -1]]
    
    # 我設計的 vertical_line_conv(垂直特徵)
    vertical_line_conv = [[1, -1], 
                         [1, -1]]
    
    return horizontal_line_conv, vertical_line_conv
    
#step3:
#把 vertical_line_conv 加到 cone_list 中，跑跑看！
def build_convolution_model(horizontal_line_conv, vertical_line_conv):
    conv_list = [horizontal_line_conv, vertical_line_conv]

    original_image = load_my_image()
    print("Original image")
    show(original_image)
    for conv in conv_list:
        filtered_image = apply_conv_to_image(conv, original_image)
        print("Output: ")
        show(filtered_image)
       
#step4:
#照片前處理 pre_process_input    
def pre_process_picture():
    hot_dog_image_dir = '/Users/sai-pc/Desktop/另外實作/transfer_learning/Code/utils/dl/hot_dog_pictures'

    hot_dog_paths = [join(hot_dog_image_dir,filename) for filename in 
                            ['hot_dog.jpg',
                             'hot_dog2.jpg']]

    not_hot_dog_image_dir = '/Users/sai-pc/Desktop/另外實作/transfer_learning/Code/utils/dl/not_hot_dog_pictures'
    not_hot_dog_paths = [join(not_hot_dog_image_dir, filename) for filename in
                            ['burger.jpg',
                             'pizza.jpg']]
    image_paths = hot_dog_paths + not_hot_dog_paths
    
    return image_paths

#step5:
#將照片 resize 成 224*224，並使用符合 Resnet50 這個模型的照片前處理(即 preprocess_input)。 
def read_and_prep_images(img_paths, img_height, img_width):
    imgs = [load_img(img_path, target_size=(img_height, img_width)) for img_path in img_paths]
    img_array = np.array([img_to_array(img) for img in imgs])
    output = preprocess_input(img_array)
    return(output)


def main():
    #step2:
    horizontal_line_conv, vertical_line_conv = design_convolution()
    
    #step3:
    build_convolution_model(horizontal_line_conv, vertical_line_conv)
    
    #step4:
    image_paths = pre_process_picture()
    
    #step5:
    image_size = 224
    img_height = image_size
    img_width = image_size
    #output = read_and_prep_images(image_paths, img_height, img_width)
    
    #step6:
    my_model = ResNet50(weights='/Users/sai-pc/Desktop/另外實作/transfer_learning/Code/utils/dl/ResNet50/resnet50_weights_tf_dim_ordering_tf_kernels.h5')
    image_data = read_and_prep_images(image_paths, img_height, img_width)
    my_preds = my_model.predict(image_data)
    
    #step7:
    #decode_predictions 把結果 show 出來
    most_likely_labels = decode_predictions(my_preds, top=2, class_list_path='/Users/sai-pc/Desktop/另外實作/transfer_learning/Code/utils/dl/result/imagenet_class_index.json')
    for i, img_path in enumerate(image_paths):
        display(Image(img_path))
        print(most_likely_labels[i])
    
if __name__ == '__main__':
    main()