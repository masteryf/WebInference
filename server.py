import socket
from utils.HttpImage import fetchImageFromHttp
from Inference.inference import GetLabel
from utils.CuttingFace import CutFace

def CycleSendText():
    host = "127.0.0.1"
    port = 6666
    tcpclient = socket.socket()
    tcpclient.connect((host, port))
    print("已连接服务端")
    while True:
        info = tcpclient.recv(1024).decode()
        data = str(info)
        data = data[2:-3]
        print(data)
        img = fetchImageFromHttp(data)
        emo = GetLabel(img)
        print(emo)
        tcpclient.send(emo.encode())

CycleSendText()
