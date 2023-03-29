# ffmedia介绍

ffmedia是一套基于Rockchip Mpp/RGA开发的视频编解码框架。支持音频aac编解码。
ffmedia一共包含以下单元

- 输入源单元 VI：
  - Camera:  支持UVC， Mipi CSI
  - RTSP Client: 支持tcp、udp和多播协议
  - File Reader：支持mkv、mp4文件读入及H264,H265裸流文件读入
- 处理单元 VP:
  - MppDec: 视频解码，支持H264,H265,MJpeg
  - MppEnc: 视频编码，支持H264,H265
  - RGA：图像缩放，裁剪，格式转换
  - AacDec: aac音频解码和播放
  - AacEnc: aac音频编码
- 输出单元 VO
  - DRM Display: 基于libdrm的显示模块
  - RTSP Server:
  - Mp4 Enmux: 编码后数据输出为Mp4格式的文件
- pybind11 pymodule.cpp
  - pymodule: 创建vi、vo、vp等的c++代码的Python绑定，以提供python调用vi、vo、vp等c++模块的python接口

## 软件框架：

整个框架采用Productor/Consumer模型，将各个单元都抽象为ModuleMedia类。

一个Productor可以有多个Consumer，一个Consumer只有一个Productor. 输入源单元没有Productor.

每个单元管理一个输出Buffer队列。各单元的输入Buffer来自其Productor的输出Buffer队列。没有Productor的单元不需要输入Buffer.

所有单元均抽象为ModuleMedia类，ModuleMedia主要有两个操作：doEnqueue, doProcess.

doEnqueue 取Productor的Buffer队列中的一个Buffer作为输入buffer, 取本单元的Buffer队列中的一个Buffer作为输出buffer。

doProcess 执行数据的处理操作，将处理后的数据写入输出Buffer.

各个单元的Buffer都统一采用MediaBuffer结构。

各个单元独立运行，流程参考ModuleMedia::work函数，从Productor的buffer队列中取一个buffer作为输入buffer，从单元自身的buffer队列中取一个buffer作为输出buffer，顺序执行doEnqueue和doProcess。

## 安装编译环境
```
apt update
apt install -y gcc g++ make cmake
apt install -y libasound2-dev libopencv-dev libdrm-dev libfdk-aac-dev
```
如果需要编译python接口库，并系统默认python是3.8，则安装3.8的软件包
```
apt install -y python3.8-dev python3.8-venv
```
如果不需要
```
sed -i 's/.*ff_pymedia*/\#&/' CMakeLists.txt
sed -i 's/add_subdirectory(pybind11)/\#&/' CMakeLists.txt
```
