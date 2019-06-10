# -*-coding: utf-8
"""
    FileName: face_recognition.py
    Author: kalentee
    E-mail: 1564920382@qq.com
    Data: 2019-5-23 11:41
    Work Speace: Visual Studio Code, Ubuntu18.04TLS, Anaconda5.1, Tensorflow1.13.1

    The last modify time:2019-5-23 11:41
"""


import face_model_test
import face_image, face_preprocess
import cv2
import os

import numpy as np

from easydict import EasyDict as edict

# 判断人脸的工作
class Recognition:
    # 这一部分的工作很简单，就是初始化模型和做人脸的匹配
    def __init__(self, conf, *path, npy=False):
        
        self.threshold = conf.threshold
        self.model = face_model_test.FaceModel(conf)

        path = path[0]
        # 加载数据库模型
        if npy:
            assert len(path) == 2
            datas_path = path[0]
            labels_path = path[1]
            self.datas = list(np.load(datas_path))
            self.labels = list(np.load(labels_path))
        else:
            assert len(path) == 1
            self.datas, self.labels = self.load_database(path[0])

    def load_database(self, img_path):

        file_list = os.listdir(img_path)
        database_datas = []
        database_labels = []

        # 设置只检测jpg， jpeg， npg格式的图片，可修改并且只遍历database_dir这一个文件夹下的文件，不遍历任何其子目录下的文件夹
        for name in file_list:
            path = os.path.join(img_path, name)

            # 递归调用load_database加载人脸库
            if os.path.isdir(path):
                print(path, "is a dir")
                data = self.load_database(path)
                if len(data) != 0:
                    database_datas.extend(data[0])
                    database_labels.extend(data[1])
                continue

            # 丢弃非jpg, jpeg, png格式外的图片，可提出来自行调整
            if not (os.path.splitext(path)[1] in ['.jpg', '.jpeg', '.png']):
                print(path, "is not a picture")
                continue

            img = cv2.imread(path)
            result = self.model.get_input(img)
            if result:
                img, _ = result
                img = self.model.get_feature(img)
                # 这里边是加载人脸库里的数据，故只可能一张图片加载一个人脸，用img[0]是降低维度
                database_datas.append(img[0])
                # 这里用多层分离分离出文件夹名称，及全部的人物类别
                database_labels.append(os.path.split(os.path.split(path)[0])[1])
        return database_datas, database_labels
    
    def judge_face(self, emb_imgs):

        name_list = []
        for img in emb_imgs:
            dist_this = []
            for data in self.datas:
                dist = np.sum(np.square(img - data))
                dist_this.append(dist)
                #print(dist)
            # 取最小的距离
            min_dist = min(dist_this)
            # 找到最小的距离在database里的编号
            number = dist_this.index(min_dist)
            print(self.labels[number], min_dist)
            if min_dist < self.threshold:
                name = self.labels[number]
            else:
                name = 'Unknow'
                print("There is an Unknow person in this house.")
            name_list.append(name)
        return name_list

    def detect_frame(self, frame):

        # 引用result接受而不是直接unpack接受是为了防止在接收时出现返回值为空的unpack错误
        result = self.model.get_input(frame)
        if result != None:
            imgs, (bboxes, _) = result
            # 提取人脸特征，作为判断是否为同一个人的标准的学习值
            imgs = self.model.get_feature(imgs)

            # 获取人脸的标签
            name_list = self.judge_face(imgs)

            # 修正bounding_boxes值为整数值，然后用作画人脸
            print("this fram is: ", name_list)
            bboxes = bboxes[:, 0:4].astype(int)
            frame = self.draw_face_box(frame, name_list, bboxes)
        
        return frame

    def draw_face_box(self, frame, boxes_name=[], boxes=[]):

        if len(boxes) != 0 and len(boxes_name) != 0:

            for name ,box in zip(boxes_name, boxes):
                cv2.rectangle(frame, (box[0],box[1]), (box[2],box[3]), (0, 255, 0), 2, 8, 0)
                cv2.putText(frame, name, (box[0],box[1]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 0, 255), thickness=2)
        
        return frame

