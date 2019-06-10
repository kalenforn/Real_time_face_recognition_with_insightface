# -*-coding: utf-8
"""
    FileName: face_recognition.py
    Author: kalentee
    E-mail: 1564920382@qq.com
    Data: 2019-5-16 11:41
    Work Speace: Visual Studio Code, Ubuntu18.04TLS, Anaconda5.1, Tensorflow1.13.1

    The last modify time:2019-5-16 11:41
"""

import face_model
import face_image, face_preprocess
import argparse
import cv2
import os

import numpy as np

database = None

def get_args():
    
    parser = argparse.ArgumentParser(description="argument of face recognition.")
    parser.add_argument('pattern', choices=['camera', 'image', 'video'], help="the pattern of this programe work. camera: using camera to detect, image: detecting an image, video: detecting a vido")
    parser.add_argument("--save-path", default="../datasets/output/", type=str, help="save path of the result")
    parser.add_argument('--database-path', default='../datasets/database/', help='database path', type=str)
    parser.add_argument('--image-size', default='112,112', help='corp image size')
    parser.add_argument('--model', default='../models/model-r50-am-lfw/model,0000', help='path to load model.')
    parser.add_argument('--ga-model', default='', help='path to load model.')
    parser.add_argument('--gpu', default=0, type=int, help='gpu id')
    parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
    parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
    parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
    
    sgroup = parser.add_argument_group("select operation")
    sgroup.add_argument('--save','-s', action='store_true', help="whether save teh result")
    sgroup.add_argument('--npy','-n', action='store_true', help="choose to load npy file or image file")

    igroup = parser.add_argument_group("operation of image detect")
    igroup.add_argument("--image-path", default="../dataset/test_image/test.jpg", type=str, help="the image path")

    vgroup = parser.add_argument_group("operation of video detect")
    vgroup.add_argument("--video-path", default="../dataset/test_video/test.mp4", type=str, help="the video path")

    ngroup = parser.add_argument_group("operation of npy file")
    ngroup.add_argument("--npy-datas", default="../datasets/npy/datas.npy", type=str, help="path of datas.npy")
    ngroup.add_argument("--npy-labels", default="../datasets/npy/labels.npy", type=str, help="path of labels.npy")

    args = parser.parse_args()

    return args

# 识别人脸
def recognition(imgs, threshold):

    global database
    labels = database[1]
    datas = database[0]
    name_list = []
    for img in imgs:
        dist_this = []
        for data in datas:
            dist = np.sum(np.square(img - data))
            dist_this.append(dist)
            #print(dist)
        # 取最小的距离
        min_dist = min(dist_this)
        # 找到最小的距离在database里的编号
        number = dist_this.index(min_dist)
        print(labels[number], min_dist)
        if min_dist < threshold:
            name = labels[number]
        else:
            name = 'Unknow'
        name_list.append(name)

    return name_list

# 画出人脸位置
def draw_face_box(img, boxes_name=[], boxes=[]):

    if len(boxes) != 0 and len(boxes_name) != 0:

        for name ,box in zip(boxes_name, boxes):
            cv2.rectangle(img, (box[0],box[1]), (box[2],box[3]), (0, 255, 0), 2, 8, 0)
            cv2.putText(img, name, (box[0],box[1]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 0, 255), thickness=2)
    
    return img

# 加载人脸数据库
def load_database(img_path, model):

    file_list = os.listdir(img_path)
    database_datas = []
    database_labels = []

    # 设置只检测jpg， jpeg， npg格式的图片，可修改并且只遍历database_dir这一个文件夹下的文件，不遍历任何其子目录下的文件夹
    for name in file_list:
        path = os.path.join(img_path, name)

        # 递归调用load_database加载人脸库
        if os.path.isdir(path):
            print(path, "is a dir")
            data = load_database(path, model)
            if len(data) != 0:
                database_datas.extend(data[0])
                database_labels.extend(data[1])
            continue

        # 丢弃非jpg, jpeg, png格式外的图片，可提出来自行调整
        if not (os.path.splitext(path)[1] in ['.jpg', '.jpeg', '.png']):
            print(path, "is not a picture")
            continue

        img = cv2.imread(path)
        result = model.get_input(img)
        if result:
            img, _ = result
            img = model.get_feature(img)
            # 这里边是加载人脸库里的数据，故只可能一张图片加载一个人脸，用img[0]是降低维度
            database_datas.append(img[0])
            # 这里用多层分离分离出文件夹名称，及全部的人物类别
            database_labels.append(os.path.split(os.path.split(path)[0])[1])

    return database_datas, database_labels

# 以npy文件形式读取文件
def load_database_by_npy(args):

    # 这里之最后加一道list是为了使其作为可迭代对象供后面使用，但是不可以用yield做返回，原因自行理解。
    datas = list(np.load(args.npy_datas))
    labels = list(np.load(args.npy_labels))

    return datas, labels

# 使用调用cv2.VideoCapture检测
# 刚开始写他单纯是为了使用摄像头检测，但是想到后面可以用于视频的检测故而做了修改，改为detect
def detect(args, model, camera=False):

    if camera:
        _open = 0
    elif args.video_path and not camera:
        if not os.path.exists(args.video_path):
            raise ValueError("Video path is not exist")
        _open = args.video_path
    else:
        raise ValueError("Parameters are not exist: args.video_path/camera only one is true!")

    print("open cap")
    cap = cv2.VideoCapture(_open)

    # 声明保存情况
    writer = None
    if args.save:
        fps = cap.get(cv2.CAP_PROP_FPS)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        writer = cv2.VideoWriter(os.path.join(args.save_path, 'output_r50.avi'), cv2.VideoWriter_fourcc(*'XVID'), fps, size)

    while cap.isOpened():
        
        # 读取图片
        _, frame = cap.read()
        # print(np.shape(frame))

        # 提取人脸和 bounding_boxes, 这里用result接收而不是用unpack方式接受是避免在检测不到人脸时返回全空而造成的unpack错误
        frame = detect_frame(args, frame, model)

        #cv2.imshow("Video", frame)

        if writer:
            frame = cv2.flip(frame, 0)
            writer.write(frame)

        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break
    
    if args.save:
        writer.release()
    cap.release()
    print("release cap")
    cv2.destroyAllWindows()
    print("destory all windows")

# 提取出公用的numpy数据检测方式
def detect_frame(args, frame, model):

    # 引用result接受而不是直接unpack接受是为了防止在接收时出现返回值为空的unpack错误
    result = model.get_input(frame)
    if result != None:
        imgs, (bboxes, _) = result
        # 提取人脸特征，作为判断是否为同一个人的标准的学习值
        imgs = model.get_feature(imgs)

        # 获取人脸的标签
        name_list = recognition(imgs, args.threshold)

        # 修正bounding_boxes值为整数值，然后用作画人脸
        print("this fram is: ", name_list)
        bboxes = bboxes[:, 0:4].astype(int)
        frame = draw_face_box(frame, name_list, bboxes)
    
    return frame

# 检测一张图片并显示   
def detect_image(args, model): 

    if not os.path.exists(args.image_path):
        raise ValueError("Image path is not exist")   

    img = cv2.imread(args.image_path)
    print("load image")
    img = detect_frame(args, img, model)

    #cv2.imshow("image", img)

    if args.save:
        cv2.imwrite(os.path.join(args.save_path, 'output.jpg'), img)

    #if cv2.waitKey(500) & 0xFF == ord('q'):pass
    
    cv2.destroyAllWindows()
    print("destroy all windows")

#工厂函数，选择工作模式
def main(args):

    # 因为只有一个函数用到了database，但是不是main函数直接调用这个函数，并且这个函数在其他函数里检测的的时候被使用，故而定义为全局变量跨函数调用
    global database
    model = face_model.FaceModel(args)
    print("load model")
    if args.npy:
        database = load_database_by_npy(args)
        print("load npy database")
    else:
        database = load_database(args.database_path, model)
        print("load image database")
    
    #选择工作模式
    if args.pattern  == 'camera':
        detect(args, model, camera=True)
    elif args.pattern == 'video':
        detect(args, model)
    elif args.pattern == 'image':
        detect_image(args, model)
    else:
        print("Error partten!")
        del database
        exit(0)
    del database
    

if __name__ == '__main__':
    args = get_args()
    main(args)




