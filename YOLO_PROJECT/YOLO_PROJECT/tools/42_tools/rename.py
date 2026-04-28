#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：step3_start_window_track.py 
@File    ：rename.py
@IDE     ：PyCharm 
@Author  ：肆十二（付费咨询QQ: 3045834499） 粉丝可享受99元调试服务
@Description  ：TODO 添加文件描述
@Date    ：2024/12/12 9:36 
'''
import os


def rename_f(src_folder = ""):
    file_names = os.listdir(src_folder)
    for file_name in file_names:
        file_path = os.path.join(src_folder, file_name)
        file_new_path = os.path.join(src_folder, file_name.split("rf.")[-1])
        os.rename(file_path, file_new_path)


if __name__ == '__main__':
    rename_f(src_folder=r"E:\danziiiiiiiiiiiii\12-batch-train\others\Yolo_test\datasets\images\train")
    rename_f(src_folder=r"E:\danziiiiiiiiiiiii\12-batch-train\others\Yolo_test\datasets\images\valid")
    rename_f(src_folder=r"E:\danziiiiiiiiiiiii\12-batch-train\others\Yolo_test\datasets\labels\valid")
    rename_f(src_folder=r"E:\danziiiiiiiiiiiii\12-batch-train\others\Yolo_test\datasets\labels\train")
    rename_f(src_folder=r"E:\danziiiiiiiiiiiii\12-batch-train\others\Yolo_test\datasets\labels\test")