import socket  
import struct  
import os      
import sys
import re
import traceback
import threading

import numpy as np
import subprocess as sp

from datetime import datetime       
from config import (DEFAULT_ADDRESS, SAVE_PATH, CONST_BUFFER_SIZE, SAVE_FPS)
from camera import VideoFrame, MyThread, write_error_logs, write_ordinary_logs

command = ['ffmpeg',
                '-y',
                '-f', 'rawvideo',
                '-vcodec','rawvideo',
                '-pix_fmt', 'bgr24',
                '-s', '640x480',
                '-r', str(10),
                '-i', '-',
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-preset', 'ultrafast',
                '-f', 'flv', 
                'rtmp://10.10.227.120:1935/live/livestream']


class Server:

    def __init__(self, address=DEFAULT_ADDRESS, savePath=SAVE_PATH, 
                    saveFrame=SAVE_FPS):
        
        self.__address = address
        self.__savePath = savePath + "/Video/"
        self.__FPS = saveFrame
        # 传输图像质量，0～100，值越高图片质量越高，但是在低速网络中值越大传输失败率也随之增大
        
        # 出使化model，方便在检测时更快速度的检测
        # self._device, self._model = load_weights()
        
        self._start_time = { "year": 2019, "month": 3, "day": 13, "hour":0, "min":0, "sec":0}
        self._initialSocket()

    def _initialSocket(self):
        
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        self._socket.bind(self.__address)
        self._socket.listen(5)
        info = "Socket Create in ip: {0}, port:{1}".format(self.__address[0], self.__address[1])
        write_ordinary_logs(info, "Run_server_log")

    def _initialConfig(self, cilent, address):

        assert isinstance(address, str)

        buffer = cilent.recv(8)
        # print("Receive data:", buffer)
        buffer = struct.unpack('ii', buffer)
        info = "Receive data from: {0}, Size:{1}".format(address, len(buffer))
        write_ordinary_logs(info, "Run_server_log", False)
        # print(buffer)

        if buffer[0] != 0:
            self.filename = self._createFile()
            resolution = (buffer[0], buffer[1])
            #self.__transSize = buffer[0]
            self._video = VideoFrame(resolution, self.__FPS)
            self._video.setVideoWriter(self.filename)
            info = "Starting transmission:\nNew writer: {}".format(self.filename)
            write_ordinary_logs(info, "Run_server_log")

    def _createFile(self):

        if not os.path.exists(self.__savePath):
            os.makedirs(self.__savePath)
            info = "Made Dir:{}".format(self.__savePath)
            write_ordinary_logs(info, "Run_server_log")

        now = datetime.now()
        day = now.strftime("%Y-%m-%d")
        thistime = now.strftime("%H-%M-%S")
        save_floder = self.__savePath + day + "/"
        if not os.path.exists(save_floder):
            os.mkdir(save_floder)
            info = "Made Dir:{}".format(save_floder)
            write_ordinary_logs(info, "Run_server_log")

        file_name = save_floder + "video_" + thistime + ".flv"
        if not os.path.isfile(file_name):
            os.mknod(file_name)
            info = "Touch File:{}".format(file_name)
            write_ordinary_logs(info, "Run_server_log")

        return file_name

    def _isCreateNewWriter(self):
        tm_year, tm_mon, tm_mday, tm_hour, tm_min, tm_sec, _, _, _ = datetime.now().timetuple()
        # 上面的flag是以一小时作为存储的判断条件，下面的flag是以一分钟作为存储判断条件
        """flag = ((self._start_time["sec"] == tm_sec and self._start_time["min"] == tm_min) and 
                    (self._start_time["hour"] + 1 == tm_hour if self._start_time["hour"] != 23 else tm_hour == 0))"""
        flag = (self._start_time["sec"] == tm_sec and 
            (self._start_time["min"] + 1 == tm_min if self._start_time["min"] != 59 else tm_min == 0))

        
        if flag:
            self._video.saveRelease()
            info = "File saved.\nFPS: {0}".format(self._video.getFrame())
            # print(self._video.getFrame())
            write_ordinary_logs(info, "Run_server_log", False)
            # print("INFO:time: {0:4d}/{1:2d}/{2:2d},{3:2d}:{4:2d}:{5:2d}, File save at :{6}\nNumber of frame is {7:04d}".format(tm_year, 
             #                       tm_mon, tm_mday, tm_hour, tm_min, tm_sec, self._file_name, self._count_frame))
            
            # print("push video", self.filename)
            # os.popen("ffmpeg -re -i %s -vcodec copy -acodec copy -f flv -y rtmp://10.10.227.120/live/livestream"%self.filename)# 2>/dev/null
            self.filename = self._createFile()

            self._video.setVideoWriter(self.filename)
            info = "New writer: {}".format(self.filename)
            write_ordinary_logs(info, "Run_server_log")
            self._start_time.update({"year": tm_year, "month": tm_mon, "day": tm_mday, 
                "hour":tm_hour, "min":tm_min, "sec":tm_sec})
            self._video.clearFrame()

    def _save_video(self, buffer):
        
        self._isCreateNewWriter()
        self._video.addFrame()
        self._video.writeVideo(buffer)
        # 管道推送到服务器
        self.pipe.stdin.write(buffer.tostring())

    def _receiveFrame(self, client, address):

        tm_year, tm_mon, tm_mday, tm_hour, tm_min, tm_sec, _, _, _ = datetime.now().timetuple()
        self._start_time.update({"year": tm_year, "month": tm_mon, "day": tm_mday, 
                "hour":tm_hour, "min":tm_min, "sec":tm_sec})
        self._start_clock = datetime.now().timetuple()

        while True:
            # 传过来的内容包含两部分，一部分是总的图片大小，另一部分就是图片数据，前4字节是图片大小
            bufSize = client.recv(4)
            bufSize = struct.unpack('i', bufSize)[0]
            buffer = b''
            # print(bufSize)
            try:
                bufferThread = MyThread(func=getOneFrame, args=(client, bufSize,))
                bufferThread.start()
                buffer = bufferThread.get_result()
                bufferThread.join()
                # 阻塞线程，需要得到完整的一张图片
                # 图片通过socket传送后的一种编解码的方式
                buffer = np.fromstring(buffer, dtype=np.uint8)
                buffer = self._video.imdecode(buffer)
                #print(type(buffer))
                #buffer = rectangle_face(buffer)
                # 还未完成的报警函数
                self._save_video(buffer)
                #self._save_video(buffer)
                self._video.showVideo(buffer)

            except Exception as e:
                write_error_logs(e, "Error_server_log")
                self._isCreateNewWriter()
             #   print(e)
              #  break
            finally:
                # 按q推出
                if self._video.waitetime(1) & 0xFF == ord('q'):
                    client.close()

    def run(self):

        while True :
            try:
                self.pipe = sp.Popen(command, stdin=sp.PIPE)
                print("Waitting connect:")
                cilent, address = self._socket.accept()
                print("Creating connect")
                self._initialConfig(cilent, address[0])
                self._receiveFrame(cilent, address)
            except Exception as e:
                write_error_logs(e, "Error_server_log")
            finally:
                #self._socket.close()
                self._video.destroyAllWindows()
                self.pipe.kill()
                if cilent:
                    cilent.close()

    def close(self):
        if self._socket != None :
            self._socket.close()
        self._video.destroyAllWindows()


def getOneFrame(client, bufSize):

    buffer = b''
    while bufSize: #循环采集一张图片
        data = client.recv(bufSize)
        bufSize -= len(data)
        buffer += data
    
    return buffer

def main():
    print(SAVE_PATH)
    try:
        os.mkdir(SAVE_PATH + "/log/")
    except:
        pass
    server = Server()
    server.run()

if __name__ == '__main__':
    main()
