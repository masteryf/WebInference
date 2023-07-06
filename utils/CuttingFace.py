import cv2

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def CutFace(img):
    face = faceCascade.detectMultiScale(img, 1.1, 5)
    size = 0
    for (x, y, w, h) in face:
        inputs = img[y:y + h, x:x + w]
        if size < h*w:
            size = h*w
            output = inputs
    return output