# -*-coding:utf-8 -*-
"""
#-------------------------------
# @Author : 肆十二
# @QQ : 3045834499 可定制毕设
#-------------------------------
# @File : step2_get_names.py
# @Description: 文件描述
# @Software : PyCharm
# @Time : 2024/2/14 13:20
#-------------------------------
"""
import os
import random
import numpy as np


annotations_foder_path = r"C:\Users\Scm97\Desktop\VOC2007\Annotations"
names = os.listdir(annotations_foder_path)
real_names = [name.split(".")[0] for name in names]
print(real_names)
random.shuffle(real_names)
print(real_names)
length = len(real_names)
split_point = int(length * 0.3)

val_names = real_names[:split_point]
train_names = real_names[split_point:]


np.savetxt('val.txt', np.array(val_names), fmt="%s", delimiter="\n")
np.savetxt('test.txt', np.array(val_names), fmt="%s", delimiter="\n")
np.savetxt('train.txt', np.array(train_names), fmt="%s", delimiter="\n")


# np.savetxt('bbbbb.txt', np.array(real_names), fmt="%s", delimiter="\n")
# E:\download\baidu\VOC2020