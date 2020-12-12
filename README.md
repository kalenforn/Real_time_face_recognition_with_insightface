# Real_time_face_recognition_with_insightface
本项目是基于python3的一个实时视频人脸识别项目（项目建立于Nvidia公司19年发布的Jetson Nano开发板上做的，配件安装请参考官方论坛），你可以用他这个项目来实现实时远程视频获取，并且在获取视频端进行识别，然后由识别端将数据发送给服务器，服务器端会实时显示，全项目依赖开源的[InsightFace项目](https://github.com/deepinsight/insightface)此处使用到的模型是mobilefacenet，你也可以到原项目地址下找其他的model使用

## Jetson Nano开发板环境搭建：
* 1、开发板的系统安装参考官方教程[Jetson Nano 开发者套件入门](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit)

你需要的外设为jetson nano的电源线、hdmi线、显示器、网线。由于开发板装系统与linux操作系统的装载差不多，而且网上教程都能搜到，所以在此仅做简述。读者可参考[Jetson-Nano开箱配置及Tensorflow安装使用](https://blog.csdn.net/dvd_sun/article/details/88975005)自行安装。
* 2、安装mxnet，由于pip相关的镜像库没有对应于jetson nano版本的mxnet

你需要到官网下载官方编译好的mxnet库，然后用pip本地安装，你也可以自己下载mxnet库到开发板编译，这里只提供官方安装mxnet方法的链接：
[CLICK](https://pan.baidu.com/s/1coo8Fgxl_mPYQBPs1082Ow)
提取码：ax1i 

### 安装mxnet的时候需要对应你的cuda版本安装(此处需要有GPU)

> pip3 install -U -r requirements.txt

服务器端需要搭建srs服务，具体配置细节请上[srs](https://github.com/ossrs/srs) 查询，建立这个服务器之前你需要在你的主机上搭建好ffmpeg命令，因为需要这个来做RTMP视频推流


在人脸识别之前你需要把自己需要识别的人脸图片放在./database/image文件夹下，根据目录下的REDME.md来做就行了

切换到./deploy文件夹下
创建人脸数据集：

> python3 create_database.py --model ../models/model-y1-test2/model,0000 -c

服务器端运行：

> python3 server.py

检测端运行：

> python3 client.py --model ../models/model-y1-test2/model,0000 -n

该项目还没完成，请慎用，后续有空我会把手机APP程序写完放上了，但由于本人想考研，所以估计得放下大概半年左右的时间，如果有大神能帮我写剩下的手机APP端rtmp显示和mqtt消息推送（或者是其他方式完成这两项任务）请你联系我，或者直接将你的程序放到这里，小弟在此感谢。

