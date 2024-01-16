import cv2
# import numpy as np
# import torch
import torchvision.transforms as transforms
from model import *
from PIL import Image
# import datetime as dt

# date = dt.date.today()
# name = f'date-{date}.mp4'
# fourcc = cv2.VideoWriter_fourcc(*'MP4V')
# out = cv2.VideoWriter(name, fourcc, 20.0, (640, 480))
model = Face_Emotion_CNN()
model.load_state_dict(torch.load('FER_trained_model.pt', map_location=lambda storage, loc: storage), strict=False)
emotion_dict = {0: 'neutral', 1: 'happiness', 2: 'surprise', 3: 'sadness',
                4: 'anger', 5: 'disgust', 6: 'fear'}
val_transform = transforms.Compose([transforms.ToTensor()])
video = cv2.VideoCapture(0)
face_detect = cv2.CascadeClassifier(
    'D:\\face_recog\\venv\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml'
)
side_face_detect = cv2.CascadeClassifier('D:\\face_recog\\venv\Lib\site-packages\cv2\data\haarcascade_profileface.xml')

while True:
    success, img = video.read()
    # out.write(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detect.detectMultiScale(gray, 1.3, 5)
    side_faces = side_face_detect.detectMultiScale(gray, 1.3, 5)
    for(x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), color=(250, 50, 255), thickness=1)

        resize_frame = cv2.resize(gray[y:y + h, x:x + w], (48, 48))
        X = resize_frame / 256
        X = Image.fromarray(X)
        X = val_transform(X).unsqueeze(0)
        with torch.no_grad():
            model.eval()
            log_ps = model.cpu()(X)
            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            # pred = emotion_dict[int(top_class.numpy())]
            pred1 = emotion_dict[top_class.item()]
        cv2.putText(img, f'{pred1}: {round(top_p.item(), 3)}', (x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, color=(255, 70, 120), thickness=2)
    # for (ex, ey, ew, eh) in side_faces:
    #     cv2.rectangle(img, (ex, ey), (ex + ew, ey + eh), (255, 0, 0))
    #     cv2.putText(img, 'My face', (ex, ey), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    #                 fontScale=1, color=(255, 70, 120), thickness=2)
    cv2.imshow('img', img)
    cv2.setWindowTitle('img', 'Face Recognition')
    # cv2.imshow('gray', gray)
    k = cv2.waitKey(1)
    if k == ord('q') or k == 27:
        break
video.release()
# out.release()
cv2.destroyAllWindows()
