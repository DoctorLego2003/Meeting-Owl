import copy
import glob
import os

import cv2
import dlib
import face_recognition
import yaml

from git.Programs.BetterTracking import main_tracking
import time

# (top, right, bottom, left)
# def yolobbox2bbox(x,y,w,h):
#     x1, y1 = x-w/2, y-h/2
#     x2, y2 = x+w/2, y+h/2
#     return x1, y1, x2, y2

def test(data):
    data = [(data[3], data[0], data[2] - data[0], data[1] - data[3])]
    return data

def load_YAML():
    # open YAML config file
    with open("MAIN_CONFIG", "r") as stream:
        try:
            # print(yaml.safe_load(stream))
            yamldata = yaml.safe_load(stream)
            YAML_DATA = copy.deepcopy(yamldata)
            yawn_thresh = YAML_DATA['yawn_threshold']
        except yaml.YAMLError as exc:
            print(exc)

        print(YAML_DATA)
        return YAML_DATA
        # for key in YAML_DATA:
        #     print(key, "=", YAML_DATA[key], "|", end=" ")
        # print("\n")



 # xml files
face_cascade = cv2.CascadeClassifier('Programs/xml/haarcascade_frontalface_default.xml')
profile_cascade = cv2.CascadeClassifier('Programs/xml/haarcascade_profileface.xml')

# face shit voor lip detection
face_model = dlib.get_frontal_face_detector()
landmark_model = dlib.shape_predictor('Programs/dat/shape_predictor_68_face_landmarks.dat')


known_face_encodings = []
known_face_names = []
zoomed = []

YAML_DATA = load_YAML()

frame = cv2.imread("./Programs/images/mauro.jpg")

# while True:
zoomed_coords = []

# print("face_reco")
# rgb_frame = frame[:, :, ::-1]
rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
face_locations = face_recognition.face_locations(rgb_frame)
# print(face_locations)
gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# print("zoomed", zoomed_coords)
main_tracking(frame, YAML_DATA, zoomed, gray_img, face_cascade, profile_cascade, face_model, landmark_model)
for persoon in zoomed:
    # print("personen", persoon)
    zoomed_coords.append(tuple(persoon[0]))
    # print("zoomed2", zoomed_coords)

face_locations2 = zoomed_coords
# face_locations = [(data[3], data[0], data[2] - data[0], data[1] - data[3]) for data in face_locations]
# NU DE BOVENSTE OPERATIE REVERSEN
# (x1, y1, x2 , y2) -> (x, y, w, h) of (y2, x1, x2-x1, y1-y2)
# dus (x, y ,w , h) -> (x1, y1, x2, y2)
# (x1, y1, x2, y2) = (data[1], , , data[0])
# x=y2, y=x1, w=x2-x1, h=y1-y2
(x, y, w, h) = face_locations2[0]
face_locations2 = [(y, h + x, w + y, x) for data in face_locations2]

print("facelocations", face_locations)
print("facelocations2", face_locations2)
# print("flipped", test(face_locations2[0]))

(x,y,w,h) = face_locations2[0]
(x2,y2,w2,h2) = face_locations[0]
# face_locations2[1:2] = face_locations2[2:1]

cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 200), 2)
cv2.putText(frame, "tracking", (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)

cv2.rectangle(frame, (x2, y2), (x2 + w2, y2 + h2), (0, 200, 200), 2)
cv2.putText(frame, "face_recog", (x2, y2 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 200, 200), 2)

# iou = bb_intersection_over_union(list(face_locations2[0]), list(face_locations[0]))
# print(iou)

frame[y-5:y+5, x-5:x+5] = (0, 0, 200)

frame[x2-5:x2+5, y2-5:y2+5] = (0, 200, 200)



cv2.imshow('Live: ', frame)
# cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

