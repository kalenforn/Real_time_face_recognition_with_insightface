import struct 
import socket
import traceback
import time
import threading
import os
import argparse

import numpy as np

from camera import VideoFrame, MyThread, write_error_logs, write_ordinary_logs
from config import DEFAULT_ADDRESS, SAVE_PATH, CONST_TIME, DEFAULT_RESOLUTION, FRAME_QUALITY
from model import Recognition

args = None

class Cilent:

    def __init__(self, address=DEFAULT_ADDRESS, resolution=DEFAULT_RESOLUTION, 
                        imgquality=FRAME_QUALITY):
        
        self.__address = address
        self.__quality = imgquality
        self._video = VideoFrame(resolution)
        self._video.setVideoCapture()
        self._sendData = None

        # 用于保存图片的三个变量
        self.__num = 0
        self.__time = [None, None]

    def _connectting(self):
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._socket.connect(self.__address)
            info = "Socket connected, address: {}".format(self.__address[0])
            write_ordinary_logs(info, "Run_client_log")
            resolution = self._video.getResolution()
            self._socket.send(struct.pack('ii', resolution[0], resolution[1]))
            # print(info)
        except Exception as e: #ConnectionRefusedError
            write_error_logs(e, "Error_client_log")
            # print(e)
        
    def _transmissionEncode(self):
        
        self._sendData = self._video.imencode(quality=self.__quality, format='.jpg')
        self._sendData = np.array(self._sendData).tobytes()
    
    def _sendFrame(self, model):
        
        # assert not (self._cilent == self.__resolution == self.__quality == None)

        """self._video.getVideoFrame()
        self._video.setImg(detect(device, model, self._video.getImg()))
        self._transmissionEncode()  
        # 初次发送一张图片大小
        self._socket.send(struct.pack('i', len(self._sendData))+self._sendData)
        info = "Send image to address:{0}, size:{1}".format(self.__address[0], len(self._sendData))
        write_ordinary_logs(info, "Run_client_log", False)"""

        while self._video.isOpened():
            try:

                if self._video:
                    self._video.getVideoFrame()
                    detectThread = MyThread(func=model.detect_frame ,args=(self._video.getImg(),))
                    detectThread.start()
                    img = detectThread.get_result()
                    self._video.setImg(img)
                    threading.Thread(target=self._transmissionEncode(), args=()).start()
                if self._sendData:
                    self._socket.send(struct.pack('i', len(self._sendData))+self._sendData)
                    # info = "Send image to address:{0}, size:{1}".format(address, len(send_data))
                    # self.__write_ordinary_logs(info, False)
            except Exception as e:
                write_error_logs(e, "Error_client_log")
                info = 'send FPS:{}'.format(self._video.getFrame())
                write_ordinary_logs(info, "Run_client_log")
                self._video.destroyAllWindows()
                return
    
    def _send(self, client, img):
        client.send(img)

    def run(self):

        global args

        self._connectting()
        model = Recognition(args, (args.npy_datas, args.npy_labels) if args.npy else (args.image_path), npy=args.npy)
        while True:
            try:
                _, _, _, _, tm_min, tm_sec, _, _, _ = time.localtime()
                self.__time = [tm_min, tm_sec]
                self._sendFrame(model)
            except Exception as e:
                write_error_logs(e, "Error_client_log")
            finally:
                self._video.destroyAllWindows()
                return

def get_args():
    
    global args
    parser = argparse.ArgumentParser(description="argument of face recognition.")
    #parser.add_argument('pattern', choices=['camera', 'image', 'video'], help="the pattern of this programe work. camera: using camera to detect, image: detecting an image, video: detecting a vido")
    parser.add_argument("--save-path", default="../datasets/output/", type=str, help="save path of the result")
    parser.add_argument('--database-path', default='../datasets/database/', help='database path', type=str)
    parser.add_argument('--image-size', default='112,112', help='corp image size')
    parser.add_argument('--model', default='../models/model-y1-test2/model,0000', help='path to load model.')
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


def main():
    get_args()
    try:
        os.mkdir(SAVE_PATH + "/log/")
    except:
        pass
    cilent = Cilent()
    cilent.run()

if __name__ == '__main__':
    main()
