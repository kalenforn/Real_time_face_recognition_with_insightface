# -*-coding: utf-8
"""
    FileName: photograph.py
    Author: kalentee
    E-mail: kalentee@hotmail.com
    Data: 2019-6-11 14:00
    Work Speace: Visual Studio Code, Ubuntu18.04TLS, Anaconda5.1, mxnet100

    The last modify time:2019-6-11 14:00
"""

import cv2
import argparse
import os
# 功能介绍：
# 使用此程序获取图片数据库，根据画面文字提示来做
def get_args():

    parse = argparse.ArgumentParser(description="Operation to take a picture")
    parse.add_argument('--name', default='test', type=str, help="Your name")
    parse.add_argument('--output-dir', default='../datasets/image', type=str, help="Picture saving dir")
    # parse.add_argument('--size', default='640,480', type=str, help="Picture's size")
    args = parse.parse_args()

    return args

def main(args):

    if not os.path.exists(args.output_dir):
        os.makedir(args.output_dir + '/' + args.name )
        print("make dir:", args.output_dir + '/' + args.name + '/')

    cam = cv2.VideoCapture(0)
    print("Open Camera.")

    num = 0
    while cam.isOpened():

        _, frame = cam.read()

        cv2.putText(frame, "S:Save image.", (10, 300), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "Q:Quite.", (10, 330), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

        flag = cv2.waitKey(1)

        cv2.imshow("Photograph", frame)

        # 按s拍照存储
        # 按q退出程序
        if flag == ord('s'):
            filename = args.output_dir + '/' + args.name + '/' + args.name + str(num) + '.jpg'
            cv2.imwrite(filename, frame)
            print("Save image: ", filename)
            num += 1
        elif flag == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    args = get_args()
    main(args)
