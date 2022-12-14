import glob
import os
import numpy

import cv2
import time
import copy
import dlib

import yaml

import multiprocessing
from multiprocessing import Process, freeze_support
import face_recognition



# met deze file ander files callen zodat het wat overzichtelijker blijft

# ------face detection------#
from Programs.BetterTracking import *
# ------face detection------#


# ------lip detection------#
# distancevorige = 0
# from Programs.LipDetection import *
from Programs.LipDetectie2 import *
# ------lip detection------#


# ------face recognition------#
from Programs.face_recog_test_mp import *
# ------face recognition------#


# ------HandGestures------#
from Programs.gestures.HandGesture import *
# ------HandGestures------#



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

def display_FPS(ptime, frame):
    # frame = img
    ctime = time.time()
    fps = int(1 / (ctime - ptime))
    ptime = ctime
    cv2.putText(frame, f'FPS:{fps}', (frame.shape[1] - 120, frame.shape[0] - 20), cv2.FONT_HERSHEY_PLAIN, 2,
                (0, 200, 0), 3)
    # cv2.putText(frame, f'FPS:{str(int(fps))}', (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
    # print(fps)
    return ptime

def load_encoding_images(images_path):
    """
    Load encoding images from path
    :param images_path:
    :return:
    """
    # Load Images
    images_path = glob.glob(os.path.join(images_path, "*.*"))

    print("{} encoding images found.".format(len(images_path)))

    # Store image encoding and names
    for img_path in images_path:
        img = cv2.imread(img_path)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Get the filename only from the initial file path.
        basename = os.path.basename(img_path)
        (filename, ext) = os.path.splitext(basename)
        # Get encoding
        img_encoding = face_recognition.face_encodings(rgb_img)[0]

        # Store file name and file encoding
        known_face_encodings.append(img_encoding)
        known_face_names.append(filename)
    print("Encoding images loaded")

def capture_vid(streamer):
    try:
        cap = cv2.VideoCapture(1)
        ret, img = cap.read()
        assert len(img) > 0
    except:
        cap = cv2.VideoCapture(0)
    video_capture = cap

    # video_capture = cv2.VideoCapture(0)
    # print("frame")
    while True:
        ret, img = video_capture.read()
        if ret:
            streamer.send(img)

def face_reco(connectie, event, lock, stream):
    while True:
        lock.acquire()
        frame = stream.recv()
        lock.release()

        # print("face_reco")
        # rgb_frame = frame[:, :, ::-1]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        print(face_locations)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        connectie.send(zip(face_locations, face_encodings))
        event.clear()
        event.wait()


def MAIN(YAML_DATA, ptime):
    streamer, stream = multiprocessing.Pipe()
    cam = Process(target=capture_vid, args=(streamer,))
    cam.start()


    face_encodings = []
    face_locations = []

    lock = multiprocessing.RLock()
    event = multiprocessing.Event()
    event.set()
    face_data_reciever, face_data_sender = multiprocessing.Pipe()
    proces = Process(target=face_reco, args=(face_data_sender, event, lock, stream,))
    proces.start()

    # lock.acquire()
    # frame = stream.recv()
    # # rgb_frame = frame[:, :, ::-1]
    # lock.release()

    face_data = []


    while True:
        lock.acquire()
        frame = stream.recv()
        lock.release()
        if not event.is_set():
            face_data = list(face_data_reciever.recv())
            # print(face_data)
            event.set()
            # print("heeft facedetectie gedaan")
            # for data in face_data:
            #     print(data)
            #     face_encodings.append(list(data[1]))
            #     face_locations.append(list(data[0]))
                # face_data = [data[0] for data in face_data]
                # face_data = [(data[3], data[0], data[2] - data[0], data[1] - data[3]) for data in face_data]

            # print(face_encodings)
            # print(face_locations)

            face_encodings = [data[1] for data in face_data]
            face_data = [data[0] for data in face_data]
            # print(face_data)

            # face_data = [(data[3], data[0], data[2] - data[0], data[1] - data[3]) for data in face_data]

            face_data = [(data[3], data[0], data[2] - data[0], data[1] - data[3]) for data in face_data]

            # print(face_data)q

        face_names = []
        for i, face_location in enumerate(face_data):
            # print(face_location)
            (x, y, w, h) = face_location

            # HIER WORDEN MATCHES VERGELEKEN MET ELKAAR
            matches = face_recognition.compare_faces(known_face_encodings, face_encodings[i])
            name = "Unknown"
            # print(matches)

            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]


            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 200), 2)
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)


        # ---------FPS------------#
        if YAML_DATA['display_FPS'] == True:
            # print("triggered")
            ptime_new = display_FPS(ptime, frame)
            ptime = ptime_new
        # ---------FPS------------#


        # # ------Tracking------#
        # #main_tracking(img, YAML_DATA, zoomed, gray_img, face_cascade, profile_cascade, distancevorige, face_model, landmark_model)
        # # main_tracking(img, YAML_DATA, zoomed, gray_img, face_cascade, profile_cascade, face_model, landmark_model)
        # # ------Tracking------#
        #
        #
        #
        #
        # # ------LipDetection------#
        # if YAML_DATA['display_lip_detection']:
        #     # OLD
        #     # main_lip_detection(img, YAML_DATA, distancevorige, gray_img, face_model, landmark_model, face_cascade)
        #     # NEW
        #     # aaaaa, bbbbb, ccccc, ddddd, eeeee
        #
        #     distancevorige, breedtemondvorige, zerocount, talklist, Talking = main_lip_detection2(img, YAML_DATA, gray_img, face_model, landmark_model, distancevorige, breedtemondvorige, zerocount, talklist, Talking)
        #
        #     # None
        #     # distancevorige = aaaaa
        #     # breedtemondvorige = bbbbb
        #     # zerocount = ccccc
        #     # talklist = ddddd
        #     # Talking = eeeee
        # # ------LipDetection------#


        # # ------HandGestures------#
        # if YAML_DATA['display_hand_gestures'] == True:
        #     main_hand_gestures(img, YAML_DATA)
        # # ------HandGestures------#


        # # ------FaceRecogntion------#
        # if YAML_DATA['display_face_recognition'] == True:
        #     print("Trying")
        #     main_face_recogntion(img, YAML_DATA)
        #     cv2.putText(img, "nice", (img.shape[1], img.shape[0] - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
        #     print("gerund")
        # # ------FaceRecogntion------#



        cv2.imshow('Live: ', frame)


        # k = cv2.waitKey(30) & 0xff
        # if k == ord('q'):
        #     break
        if cv2.waitKey(30) & 0xff == ord('q') or cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()




if __name__ == "__main__":
    ptime = 0

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

    known_face_encodings = []
    known_face_names = []



    YAML_DATA = load_YAML()

    load_encoding_images("Programs/images/")
    # print(known_face_names)



    # cap = cv2.VideoCapture(0)
    # ret, img = cap.read()
    # frame = img
    try:
        cap = cv2.VideoCapture(1)
        ret, img = cap.read()
        assert len(img) > 0
    except:
        cap = cv2.VideoCapture(0)

    # face_locations = face_recognition.face_locations(frame)
    # face_encodings = face_recognition.face_encodings(frame, face_locations)
    # cap.release()

    MAIN(YAML_DATA, ptime)
