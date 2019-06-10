import os

path = os.getcwd()
os.chdir(os.path.pardir)
# save path 
SAVE_PATH = os.getcwd()
os.chdir(path)

# video storage frame
# 用tiny模型就是18帧每秒，用spp就只能用8帧每秒了
SAVE_FPS = 10

# video transmission format
FRAME_QUALITY = 80

# default resolution
DEFAULT_RESOLUTION = (640,480)

# default address of raspberry
# 192.168.1.103
# 10.10.227.119
DEFAULT_ADDRESS = ("10.10.227.120", 7999)

# picture size
CONST_BUFFER_SIZE = DEFAULT_RESOLUTION[0] * DEFAULT_RESOLUTION[1]

# const time to save picture
CONST_TIME = 8

