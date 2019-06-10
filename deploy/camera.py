import cv2
import time
import traceback
import threading

from config import DEFAULT_RESOLUTION, SAVE_FPS, SAVE_PATH

SAVE_FORMAT = cv2.VideoWriter_fourcc(*'FLV1')

class VideoFrame:

    def __init__(self, resolution=DEFAULT_RESOLUTION, saveFPS=SAVE_FPS, saveFormat=SAVE_FORMAT):

        # initial all default video config parameter
        self.__resolution = resolution
        self.__saveFPS = saveFPS
        self.__saveFormat = saveFormat

        # initial other value
        self._count_frame = 0
        self._img = []
        self._cap = None
        self._writer = None

    def addFrame(self):
        self._count_frame += 1
    
    def clearFrame(self):
        self._count_frame = 0
    
    # 外部获取帧数的接口
    def getFrame(self):
        return self._count_frame
    
    # 获取帧的接口
    def getImg(self):
        return self._img

    def setImg(self, img):
        self._img = img
    
    def getResolution(self):
        return self.__resolution

    # 初始化摄像头，在server端可不用
    def setVideoCapture(self, num=0):
        self._cap = cv2.VideoCapture(num)
        #fps = self._cap.get(cv2.CAP_PROP_FPS)
        #print("fps", fps)
    
    # 初始化写的单位，在client端可以不用
    def setVideoWriter(self, savefile):
        self._writer = cv2.VideoWriter(savefile, self.__saveFormat, self.__saveFPS , self.__resolution)
    
    def imdecode(self, buffer, flags=1):
        return cv2.imdecode(buffer, flags)
    
    # 编码，只负责编img变量的函数，在socket传输中还需要加工
    def imencode(self, quality=12, format='.jpg'):

        assert len(self._img) != 0

        params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        img = cv2.resize(self._img , self.__resolution)
        return cv2.imencode(format, img, params)[1]

    # 获取视频帧
    def getVideoFrame(self):

        assert self._cap != None

        # 帧存储在保护变量img里
        self._img = self._cap.read()[1]
        # 帧数加1
        # print(self._img)
        self.addFrame()
        # threading.Thread(target=self.rectangleFace(), args=()).start()
        #return self.getImg()

    def writeVideo(self, buffer):

        assert self._writer != None

        _, _, _, tm_hour, tm_min, tm_sec, _, _, _ = time.localtime()
        cv2.putText(buffer, str(tm_hour) + ":" + str(tm_min) + ":" + str(tm_sec), (10, 20), 
                        cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1, cv2.LINE_AA)
        self._writer.write(buffer)
        self.addFrame()
    
    def savePicture(self, filename):
        cv2.imwrite(filename, self._img)

    # 释放writer,保存视频
    def saveRelease(self):

        assert self._writer != None

        self._writer.release()
        self._writer = None

    def showVideo(self, buffer):
        cv2.imshow("Video", buffer)

    # 防止程序意外bug使数据丢失
    def destroyAllWindows(self):
        if self._cap != None:
            self._cap.release()
        if self._writer != None:
            self._writer.release()
        cv2.destroyAllWindows()

    def waitetime(self, time):
        return cv2.waitKey(time)

    
    def isOpened(self):
        return self._cap.isOpened()
        """
    def rectangleFace(self):

        gray = cv2.cvtColor(self._img, cv2.COLOR_BGR2GRAY)
        color = (0, 255, 0)
        path_name = "haarcascade_frontalface_default.xml" #"haarcascade_frontalface_default.xml"
        faceCascade = cv2.CascadeClassifier(path_name)

        faceRects = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))

        if len(faceRects):
            for faceRect in faceRects:

                x, y, w, h = faceRect
                cv2.rectangle(self._img, (x, y), (x + h, y + w), color, 2)
"""

class MyThread(threading.Thread):

    def __init__(self, func, args=()):

        super(MyThread,self).__init__()
        self.func = func
        self.args = args
        self.result = self.func(*self.args)    
    
    def get_result(self):
        try:
            return self.result
        except Exception:
            return None


def write_error_logs(e, log_name):

    assert isinstance(log_name, str)

    tm_year, tm_mon, tm_mday, tm_hour, tm_min, tm_sec, _, _, _ = time.localtime()
    with open(SAVE_PATH + "/log/" + log_name, 'a+') as f:
        f.writelines("\n*********************************************************************\n")
        f.write("Error:time:{0:4d}/{1:2d}/{2:2d},{3:2d}:{4:2d}:{5:2d}\\\\\n:{6}".format(tm_year, 
                    tm_mon, tm_mday, tm_hour, tm_min, tm_sec, traceback.format_exc()))
        f.writelines("*********************************************************************\n")

def write_ordinary_logs(info, log_name, head=True):
        
        assert isinstance(info, str)
        assert isinstance(head, bool)
        assert isinstance(log_name, str)

        tm_year, tm_mon, tm_mday, tm_hour, tm_min, tm_sec, _, _, _ = time.localtime()
        with open(SAVE_PATH + '/log/' + log_name, 'a+') as f:
            if head :
                f.writelines("\n*********************************************************************\n")
            f.write("INFO:time:{0:4d}/{1:2d}/{2:2d},{3:2d}:{4:2d}:{5:2d}\\\\{6}".format(tm_year, 
                        tm_mon, tm_mday, tm_hour, tm_min, tm_sec, info + "\n"))
