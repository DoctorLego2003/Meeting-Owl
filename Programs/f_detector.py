import cv2
import numpy as np
import config as cfg

def detect(img, cascade):
    #rects,_,confidence = cascade.detectMultiScale3(img, scaleFactor=1.25, minNeighbors=4, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE, outputRejectLevels = True)
    rects = cascade.detectMultiScale(img, 1.25, 4)
    if len(rects) == 0:
        return (),()
    confidence = None
    return rects,confidence


def convert_rightbox(img,box_right):
    res = np.array([])
    _,x_max = img.shape
    for box_ in box_right:
        box = np.copy(box_)
        box[0] = x_max-box_[2]
        box[2] = x_max-box_[0]
        if res.size == 0:
            res = np.expand_dims(box,axis=0)
        else:
            res = np.vstack((res,box))
    return res


class detect_face_orientation():
    def __init__(self):
        # crear el detector de rostros frontal
        self.detect_frontal_face = cv2.CascadeClassifier(cfg.detect_frontal_face)
        # crear el detector de perfil rostros
        self.detect_perfil_face = cv2.CascadeClassifier(cfg.detect_perfil_face)
    def face_orientation(self,gray):
        # frontal_face
        box_frontal,w_frontal = detect(gray,self.detect_frontal_face)
        if len(box_frontal)==0:
            box_frontal = []
        # left_face
        box_left, w_left = detect(gray,self.detect_perfil_face)
        if len(box_left)==0:
            box_left = []
        # right_face
        gray_flipped = cv2.flip(gray, 1)
        box_right, w_right = detect(gray_flipped,self.detect_perfil_face)
        if len(box_right)==0:
            box_right = []
        else:
            box_right = convert_rightbox(gray,box_right)

        if len(box_frontal) != 0:
            box_left = []
            box_right = []



        boxes = list(box_frontal)+list(box_left)+list(box_right)
        return boxes

