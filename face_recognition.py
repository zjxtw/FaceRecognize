"""
作者: ryez
日期: 2021年08月01日
"""
import os
import common as common
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import cv2
from model.DBFace import DBFace
from main import detect
import time

# 输入、输出图像路径
sources = 'E:\\Data\\CCF_Gaze\\data\\training_data'
target = 'E:\\Data\\CCF_Gaze\\text'

training_dirs = os.listdir(sources)
source_training_dirs = [os.path.join(sources, k) for k in training_dirs]
target_training_dirs = [os.path.join(target, k) for k in training_dirs]

source_training_data = [os.listdir(k) for k in source_training_dirs]
# 构造检测器
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
dbface = DBFace()
dbface.eval()
HAS_CUDA = torch.cuda.is_available()
if HAS_CUDA:
    dbface.cuda()

dbface.load("model/dbface.pth")
# 检测人脸
for i, files in enumerate(source_training_data):
    m = 0
    if i == 0:
        print('processing face-images-label: ' + source_training_dirs[i] + ' ... ', end="")
        start_time = time.time()
        if not os.path.exists(target_training_dirs[i]):
            os.makedirs(target_training_dirs[i])
        files.sort(key=lambda s: len(s))  # 排序操作
        # 标签文件
        out_txt = open(os.path.join(target_training_dirs[i], 'label_camera.txt'), 'w')
        for file in files:
            input_path = os.path.join(source_training_dirs[i], file)
            # if file.endswith('jpg'):
            #     out_path = os.path.join(target_training_dirs[i], file)
            #
            #     frame = cv2.imread(input_path)
            #     objs = detect(dbface, frame)
            #     if objs:
            #         target_obj = objs[0]
            #         for obj in objs:
            #             l1, r1 = int(target_obj.box[0]) if int(target_obj.box[0]) > 0 else 0, int(
            #                 target_obj.box[2]) if int(
            #                 target_obj.box[2]) > 0 else 0
            #             l2, r2 = int(obj.box[0]) if int(obj.box[0]) > 0 else 0, int(obj.box[2]) if int(
            #                 obj.box[2]) > 0 else 0
            #             if (r1 - l1) < (r2 - l2):
            #                 target_obj = obj
            #         out_image = common.drawbbox(frame, target_obj)
            #         cv2.imwrite(out_path, out_image)
            #     else:
            #         cv2.imwrite(out_path, frame)
            if file.endswith('gaze.txt'):
                if m == int(len(files)/3)-1:
                    input_txt = open(input_path)
                    content = input_txt.readline().replace('\n', '')
                    out_txt.write(content)
                else:
                    input_txt = open(input_path)
                    content = input_txt.readline()
                    out_txt.write(content)
                    m = m+1

        #lines = out_txt.readlines()
        #last_line = lines[-1]

        out_txt.close()
        end_time = time.time()
        print('finished [ running time ' + str(int(end_time - start_time)) + ' s ]')
