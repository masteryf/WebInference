# 运行Server.py来启动服务器

server.py是一个在线预测服务器的简单例子，以下是如何更改的说明



## 预测方法：

导入：

```python
from Inference.inference import GetLabel
```

使用：
```python
emo = GetLabel(img)
```
其中，参数为需要预测的图片，返回值为标签（一个整数）

在这个例子中应该是一张裁切好的人脸

## 从网络获取图片方法：

导入：

```python
from utils.HttpImage import fetchImageFromHttp
```

使用：
```python
img = fetchImageFromHttp("http://your.photos.url")
```
其中，参数为图片地址，返回值为图片（opencv图片）

## 从图片中裁切人脸方法：

不建议使用，建议在前端完成

导入：

```python
from utils.CuttingFace import CutFace
```

使用：
```python
img = CutFace(img)
```
其中，输入输出皆为图片（opencv图片）