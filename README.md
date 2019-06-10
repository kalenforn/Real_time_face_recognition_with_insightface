# video_face_recognition_python_mxnet_insightface
本项目是基于python3的一个实时视频人脸识别项目，你可以用他这个项目来实现实时远程视频获取，并且在获取端进行识别，然后由识别端将数据发送给服务器，服务器端会实时显示，全项目依赖开源的InsightFace项目https://github.com/deepinsight/insightface  此处使用到的模型是mobilefacenet，你也可以到原项目地址下找其他的model使用


服务器端需要搭建srs服务，具体配置细节请上 https://github.com/ossrs/srs 查询，建立这个服务器之前你需要在你的主机上搭建好ffmpeg命令，因为需要这个来做RTMP视频推流


在人脸识别之前你需要把自己需要识别的人脸图片放在./database/image文件夹下，根据目录下的REDME.md来做就行了

切换到./deploy文件夹下
创建人脸数据集：

`python3 create_database.py --model ../models/model-y1-test2/model,0000 -c`

服务器端运行：

`python3 server.py`

检测端运行：

`python3 client.py --model ../models/model-y1-test2/model,0000 -n`
