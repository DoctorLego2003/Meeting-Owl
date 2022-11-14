import cv2
import time
import copy
import numpy
prev = []
zoomed = []
YAML_DATA = {}
face_cascade = cv2.CascadeClassifier('Programs/xml/haarcascade_frontalface_default.xml')
profile_cascade = cv2.CascadeClassifier('Programs/xml/haarcascade_profileface.xml')

# open YAML config file
import yaml
with open("MAIN_CONFIG", "r") as stream:
    try:
        # print(yaml.safe_load(stream))
        yamldata = yaml.safe_load(stream)
        YAML_DATA = copy.deepcopy(yamldata)
    except yaml.YAMLError as exc:
        print(exc)

print(YAML_DATA)





# met deze file ander files callen zodat het wat overzichtelijker blijft

# face detection
from Programs.BetterTracking import *


# lip detection

# face recognition

# ---------FPS------------#
ptime = 0
def display_FPS(ptime):
    frame = img
    ctime = time.time()
    fps = int(1 / (ctime - ptime))
    ptime = ctime
    cv2.putText(frame, f'FPS:{fps}', (frame.shape[1] - 120, frame.shape[0] - 20), cv2.FONT_HERSHEY_PLAIN, 2,
                (0, 200, 0), 3)
    return ptime
# ---------FPS------------#



try:
    cap = cv2.VideoCapture(1)
    ret, img = cap.read()
    assert len(img) > 0
except:
    cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #faces = face_cascade.detectMultiScale(gray_img, 1.25, 4)
    faces = detect_face_orientation(gray_img, face_cascade, profile_cascade)
    print('faces:', faces, len(faces))
    faces = track(faces, zoomed)


    # ---------FPS------------#
    if YAML_DATA['display_FPS'] == True:
        ptime_new = display_FPS(ptime)
        ptime = ptime_new
    # ---------FPS------------#


    for i in range(len(faces)):
        if len(faces) > len(zoomed):
            zoomed.append([[], []])
        (x, y, w, h) = faces[i]

        # -----SHOW RECTANGLE-----#
        if YAML_DATA['display_face_detection'] == True:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
        # -----SHOW RECTANGLE-----#

        # ------SHOW ZOOMED------#
        if YAML_DATA['display_face_detection_zoomed'] == True:
            rec_gray = gray_img[y:y + h, x:x + w]
            rec_color = img[y:y + h, x:x + w]
            htot = 3 * h//2
            wtot = 3 * w//2
            h2 = (htot - h)//2
            w2 = (wtot - w)//2
            if y < h2:
                h2 = y
            if x + w2 + w > len(gray_img[0]):
                w2 = len(gray_img[0]) - w - x
            if x < w2:
                w2 = x
            if y + h + h2 > len(gray_img):
                h2 = len(gray_img) - y - h
            zoomed[i][0] = faces[i]
            #print(zoomed)

            #print(len(gray_img))
            #print(len(gray_img[0]))

            head_frame = cv2.resize(img[y - h2: y + h2 + h, x - w2: x + w2 + w], (400, 400))
            cv2.imshow('Zoom in ' + str(i + 1), head_frame)
        # ------SHOW ZOOMED------#

        # cv2.resizeWindow('Zoom in ' + str(i+1), 400, 400)
        # cv2.resizeWindow('Zoom in ' + str(i + 1), 325, 325)

    cv2.imshow('Live: ', img)
    # ------SHOW ZOOMED------#
    if YAML_DATA['display_face_detection_zoomed'] == True:
        removed = []
        for i in range(len(zoomed)):
            # print(zoomed[i][0])
            # print(zoomed[i][1])
            if zoomed[i][0] == zoomed[i][1]:
                cv2.destroyWindow('Zoom in ' + str(i + 1))
                removed.append(zoomed[i])
            else:
                zoomed[i][1] = zoomed[i][0]
        for x in removed:
            zoomed.remove(x)
    # ------SHOW ZOOMED------#


    k = cv2.waitKey(30) & 0xff
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
