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
import os.path as osp
import numpy as np
import cv2


def cv_imread_chinese(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    return cv_img


def folder_rename(src_folder_path, target_folder_path):
    os.makedirs(target_folder_path, exist_ok=True)
    file_names = os.listdir(src_folder_path)
    for i, file_name in enumerate(file_names):
        print("{}:{}".format(i, file_name))
        src_name= osp.join(src_folder_path, file_name)
        src_img = cv_imread_chinese(src_name)
        target_path = osp.join(target_folder_path, "yolo_data_{}.jpg".format(i))
        cv2.imwrite( target_path,src_img )
        # os.rename(src_name, target_name)

if __name__ == '__main__':
    # 脚本应该生成在一个新的目录下，防止出错
    folder_rename(r"G:\Upppppdate\AAAAA-standard-code\yolo11\yolo11-up\42_tools\data_process\test_data\中文路径", r"G:\Upppppdate\AAAAA-standard-code\yolo11\yolo11-up\42_tools\data_process\test_data\english_path")
    # folder_rename(r"G:\AAA-projects\update\yolov8\dataset_shudian\xml")