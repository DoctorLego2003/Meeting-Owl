import cv2
import numpy

mouth_cascade = cv2.CascadeClassifier(r'./xml/haarcascade_mouth.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, frames = cap.read()
    gray_img = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
    #faces = face_cascade.detectMultiScale(gray_img, 1.25, 4)
    mouth = mouth_cascade.detectMultiScale(gray_img, 1.25, 4)
    print(mouth)

    for (x, y, w, h) in mouth:
        cv2.rectangle(frames, (x, y), (x + w, y + h), (255, 255, 0), 2)
    cv2.imshow('Video', frames)

cap.release()
cv2.destroyAllWindows()