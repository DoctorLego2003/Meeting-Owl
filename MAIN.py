import cv2
import time
import copy
import dlib


prev = []
zoomed = []
YAML_DATA = {}


# xml files
face_cascade = cv2.CascadeClassifier('Programs/xml/haarcascade_frontalface_default.xml')
profile_cascade = cv2.CascadeClassifier('Programs/xml/haarcascade_profileface.xml')


# face shit voor lip detection
face_model = dlib.get_frontal_face_detector()
landmark_model = dlib.shape_predictor('Programs/dat/shape_predictor_68_face_landmarks.dat')


distancevorige = 0
breedtemondvorige = 1
zerocount = 0
talklist = 0
Talking = False


# open YAML config file
import yaml
with open("MAIN_CONFIG", "r") as stream:
    try:
        # print(yaml.safe_load(stream))
        yamldata = yaml.safe_load(stream)
        YAML_DATA = copy.deepcopy(yamldata)
        yawn_thresh = YAML_DATA['yawn_threshold']
    except yaml.YAMLError as exc:
        print(exc)


print(YAML_DATA)
# for key in YAML_DATA:
#     print(key, "=", YAML_DATA[key], "|", end=" ")
# print("\n")



# met deze file ander files callen zodat het wat overzichtelijker blijft

# ------face detection------#
from Programs.BetterTracking import *
# ------face detection------#


# ------lip detection------#
distancevorige = 0
from Programs.LipDetection import *
from Programs.LipDetectie2 import *
# ------lip detection------#


# ------face recognition------#
from Programs.face_recog_test_mp import *
# ------face recognition------#


# ------HandGestures------#
from Programs.gestures.HandGesture import *
# ------HandGestures------#



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


# -----MAIN------#

try:
    cap = cv2.VideoCapture(0)
    ret, img = cap.read()
    assert len(img) > 0
except:
    cap = cv2.VideoCapture(1)

# cap= cv2.VideoCapture('Programs/test_footage/IMG_8235.mp4')

while True:
    ret, img = cap.read()
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #faces = face_cascade.detectMultiScale(gray_img, 1.25, 4)
    # faces = detect_face_orientation(gray_img, face_cascade, profile_cascade)
    # print('faces:', faces, len(faces))
    # faces = track(faces, zoomed)

    # ---------FPS------------#
    if YAML_DATA['display_FPS'] == True:
        ptime_new = display_FPS(ptime)
        ptime = ptime_new
    # ---------FPS------------#

    # ------Tracking------#
    #main_tracking(img, YAML_DATA, zoomed, gray_img, face_cascade, profile_cascade, distancevorige, face_model, landmark_model)
    main_tracking(img, YAML_DATA, zoomed, gray_img, face_cascade, profile_cascade, face_model, landmark_model)
    # ------Tracking------#

    # ------LipDetection------#
    #if YAML_DATA['display_lip_detection']:
        # OLD
        # main_lip_detection(img, YAML_DATA, distancevorige, gray_img, face_model, landmark_model, face_cascade)
        # NEW
        #distancevorige, breedtemondvorige, zerocount, talklist, Talking = main_lip_detection2(img, YAML_DATA, gray_img, face_model, landmark_model, distancevorige, breedtemondvorige, zerocount, talklist, Talking)
    # ------LipDetection------#

    # ------HandGestures------#
    # if YAML_DATA['display_hand_gestures'] == True:
    #     main_hand_gestures(img, YAML_DATA)
    # ------HandGestures------#


    # # ------FaceRecogntion------#
    # if YAML_DATA['display_face_recognition'] == True:
    #     print("Trying")
    #     main_face_recogntion(img, YAML_DATA)
    #     cv2.putText(img, "nice", (img.shape[1], img.shape[0] - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
    #     print("gerund")
    # # ------FaceRecogntion------#



    cv2.imshow('Live: ', img)


    k = cv2.waitKey(30) & 0xff
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
