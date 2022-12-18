import copy

import cv2
import numpy
# prev = []
# zoomed = []
# face_cascade = cv2.CascadeClassifier(r'./xml/haarcascade_frontalface_default.xml')
# profile_cascade = cv2.CascadeClassifier(r'./xml/haarcascade_profileface.xml')

# from .LipDetection import *
from .gestures.HandGesture import *
from .face_recog_test_mp import *
from .LipDetectie2 import *

from .Organising import *


def intersection(a,b):
  x = max(a[0], b[0])
  y = max(a[1], b[1])
  w = min(a[0]+a[2], b[0]+b[2]) - x
  h = min(a[1]+a[3], b[1]+b[3]) - y
  if w<0 or h<0: return False
  return True

def detect_face_orientation(gray_img, face_cascade, profile_cascade):
    faces = []
    front_faces = face_cascade.detectMultiScale(gray_img, 1.25, 4)
    left_profile = profile_cascade.detectMultiScale(gray_img, 1.3, 3)
    gray_flipped = cv2.flip(gray_img, 1)
    """
    faces = profile_cascade.detectMultiScale(gray_flipped, 1.3, 4)
    if len(faces) != 0:
        h, w = img.shape
        x = faces[0][0]
        new_x = h - x -1
        print(w, x, new_x)
        faces[0][0] = new_x
    """
    right_profile = profile_cascade.detectMultiScale(gray_flipped, 1.3, 3)

    if type(front_faces) is not tuple:
        front_faces = front_faces.tolist()
        #print(front_faces)
    if type(left_profile) is not tuple:
        left_profile = left_profile.tolist()
        #print(left_profile)
    if type(right_profile) is not tuple:
        right_profile = right_profile.tolist()
        #print(right_profile)

    if len(front_faces) != 0:
        for face in front_faces:
            faces.append(face)
    if len(left_profile) != 0:
        for face in left_profile:
            if len(faces) == 0:
                faces.append(face)
            else:
                check = True
                for second_face in faces:
                    if intersection(face, second_face):
                        check = False
                if check:
                    faces.append(face)
    if len(right_profile) != 0:
        for face in right_profile:
            w, h = gray_img.shape
            x = face[0]
            new_x = w - x - 1
            face[0] = new_x
            if len(faces) == 0:
                faces.append(face)
            else:
                check = True
                for second_face in faces:
                    if intersection(face,second_face):
                        check = False
                if check:
                    faces.append(face)
    #for face in faces:
    #    (x, y, w, h) = face
    #    assert x >= 0
    #    assert y >= 0
    #    assert w >= 0
    #    assert h >= 0
    check_for_double_faces(faces)
    check_for_empty_faces(faces)
    return faces

def distance(point_one, point_two):
    if point_one == []:
        return float('inf')
    [x1, y1, w0, h0] = point_one
    [x2, y2, w1, h1] = point_two
    dist = ((x2 - x1)**2 + (y2 - y1)**2)
    return dist

def track(faces, zoomed, YAML_DATA):
    if len(zoomed) == 0:
        return faces
    #print('zoomed: ', zoomed)
    dist = []
    for i in range(len(zoomed)):
        dist.append([])
        for punt in faces:
            dist[i].append(distance(zoomed[i][0], punt))
    # print('dist:', dist)
    #calculating lowest distance per point
    min_ind = []
    for i in range(len(dist)):
        min_ind.append([])
        if len(dist[i]) != 0:
            mini = min(dist[i])
            if mini <= YAML_DATA['minimum_distance']:
                counter = 0
                for thing in dist[i]:
                    if thing == mini:
                        counter += 1
                if counter > 1:
                    for j in range(counter):
                        index_new = dist[i].index(mini)
                        min_ind[i].append(index_new)
                        dist[i][index_new] = -1
                else:
                    min_ind[i].append(dist[i].index(mini))
    # print('min_ind:', min_ind)
    ## better filtering for equal lengths between faces

    i = 0
    while i < len(min_ind):
        if len(min_ind[i]) > 1:
            for point in min_ind[i]:
                if [point] in min_ind and len(min_ind[i]) > 1:
                    min_ind[i].remove(point)
        i += 1
    #returning a reorganised faces
    # print('faces: ', faces)
    new_faces = []
    missing_index = []
    for i in range(len(min_ind)):
        index = min_ind[i]
        # print('index:', index)
        if index != []:
            new_faces.append(faces[index[0]])
        else:
            missing_index.append(i)
    if (len(new_faces) > 0) and (len(missing_index) > 0):
        #print('missing_index:', missing_index)
        for i in range(len(missing_index)):
            min_ind.append(missing_index[i])
            if i >= len(new_faces):
                new_faces.append(faces[missing_index[i][0]])
            elif i < len(zoomed):
                new_faces.insert(missing_index[i], zoomed[missing_index[i]][0])
            else:
                new_faces.insert(missing_index[i], zoomed[missing_index[i]][0])

    if len(new_faces) < len(faces):
        all_index = [x for x in range(len(faces))]
        missing_index = list(filter(lambda x: [x] not in min_ind, all_index))
        # print('missing_index:', missing_index)
        for i in range(len(missing_index)):
            new_faces.append(faces[missing_index[i]])
        #print('new:', new_faces)
    #print('faces: ', faces)
    #print('new_faces: ', new_faces)
    # print('faces1:', faces)
    check_for_double_faces(new_faces)
    #print('faces2:', faces)
    check_for_empty_faces(new_faces)
    # print('faces3:', new_faces)
    #print('zoomed', zoomed)
    #print('faces4:', new_faces)
    # print('---------------------')
    return new_faces

def check_for_doubles(zoomed):
    for i in range(len(zoomed)):
        j = i + 1
        while j < len(zoomed):
            if zoomed[i][0] == zoomed[j][0] and zoomed[i][1] == zoomed[j][1]:
                zoomed.remove(zoomed[j])
            else:
                j += 1
def check_for_double_faces(faces):
    i = 0
    while i < len(faces):
        j = i + 1
        while j < len(faces):
            if faces[i] == faces[j]:
                faces.remove(faces[j])
            else:
                j += 1
        i += 1

def check_for_empty_faces(faces):
    i = 0
    while i < len(faces):
        if len(faces[i]) == 0:
            faces.remove(faces[i])
        else:
            i += 1

def check_for_empty(zoomed):
    i = 0
    while i < len(zoomed):
        if len(zoomed[i][0]) != 4:
            zoomed.remove(zoomed[i])
        else:
            i += 1

def make_frame(img, face):
    (x, y, w, h) = face
    htot = 3 * h // 2
    wtot = 3 * w // 2
    h2 = (htot - h) // 2
    w2 = (wtot - w) // 2
    x_start = x - w2
    y_start = y - h2
    x_eind = x + w + w2
    y_eind = y + h + h2
    if x_start < 0:
        if x_eind - x_start < len(img[0]):
            x_eind = x_eind - x_start
        else:
            x_eind = len(img[0]) - 1
        x_start = 0
    if x_eind > len(img[0]):
        if x_start + len(img[0]) - x_eind > 0:
            x_start = x_start + len(img[0]) - x_eind
        else:
            x_start = 0
        x_eind = len(img[0])
    if y_start < 0:
        if y_eind - y_start < len(img):
            y_eind = y_eind - y_start
        else:
            y_eind = len(img) - 1
        y_start = 0
    if y_eind > len(img):
        if y_start + len(img) - y_eind > 0:
            y_start = y_start + len(img) - y_eind
        else:
            y_start = 0
        y_eind = len(img)
    return x_start, x_eind, y_start, y_eind

# while True:
#img, YAML_DATA, zoomed, gray_img, face_cascade, profile_cascade, distancevorige, face_model, landmark_model
def main_tracking(img, YAML_DATA, zoomed, gray_img, face_cascade, profile_cascade, face_model, landmark_model, testconn2sender): #+ testconn2sender
#    ret, img = cap.read()
#    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#    faces = face_cascade.detectMultiScale(gray_img, 1.25, 4)
    faces = detect_face_orientation(gray_img, face_cascade, profile_cascade)
    # print('faces:', faces, len(faces))
    faces = track(faces, zoomed, YAML_DATA)
    show = []
    #check_for_doubles(zoomed)
    #check_for_empty(zoomed)

    i = 0
    while 0 <= i < len(faces):
        if len(faces) > len(zoomed):
            zoomed.append([[], 3, [], str(), False, []])
        # -----SHOW RECTANGLE-----#

        # -----SHOW RECTANGLE-----#
        #     # ------FaceRecogntion------#
        #     if YAML_DATA['display_face_recognition'] == True:
        #         main_face_recogntion(img, YAML_DATA)
        #         cv2.putText(img, "69 nice", (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
        #
        # # ------FaceRecogntion------#

        # ------SHOW ZOOMED------#
        #rec_gray = gray_img[y:y + h, x:x + w]
        #rec_color = img[y:y + h, x:x + w]

        # print('prev zoomed:', zoomed)
        # print('prev faces:', faces)
        changed = False
        if zoomed[i][0] != faces[i]:
            c = 0.1
            if len(zoomed[i][0]) == len(faces[i]) and len(faces[i]) != 0:
                for j in range(4):
                    number = c * faces[i][j] + (1 - c) * zoomed[i][0][j]
                    if int(number) < int(number + 0.5):
                        faces[i][j] = int(number + 1)
                    else:
                        faces[i][j] = int(number)
                changed = True
        #cv2.imshow('faces' + str(i + 1), faces[i])
        if zoomed[i][0] != faces[i] or changed:
            if zoomed[i][1] < YAML_DATA['tracking_treshhold_high']:
                zoomed[i][1] += 1
        elif zoomed[i][1] > 0:
            zoomed[i][1] -= 1

        x_start, x_eind, y_start, y_eind = make_frame(img, faces[i])
        if YAML_DATA['display_face_detection']:
            (x, y, w, h) = faces[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)

        zoomed[i][0] = faces[i]

        if zoomed[i][1] == 0:
            zoomed.remove(zoomed[i])
            faces.remove(faces[i])
            i -= 1

        elif (zoomed[i][1] >= YAML_DATA['tracking_treshhold_low']) and (zoomed[i][1] <= YAML_DATA['tracking_treshhold_high']):
            head_frame = img[y_start: y_eind, x_start: x_eind]

            head_frame = cv2.resize(head_frame, (400, 400))
            # ------HandGestures------#
            if YAML_DATA['display_hand_gestures']:
                a = YAML_DATA['extra_width']
                b = YAML_DATA['extra_height']
                x_end = int(x + (1+a)*w)
                y_end = int(y + (1+b)*h)
                x_new = x - w
                if x_end >= len(img[0]):
                    x_end = len(img[0]) - 1
                    x_new = min(len(img[0]) - int((2*a+1)*w), x_new)
                if x_new < 0:
                    x_new = 1
                    if (2*a+1)*w < len(img[0]):
                        x_end = int((2*a+1)*w)
                    else:
                        x_end = len(img[0])
                if y_end > len(img):
                    y_end = len(img) - 1
                hand_frame = img[y:y_end, x_new:x_end]
                gesture = main_hand_gestures(hand_frame, YAML_DATA)
                if gesture != None:
                    zoomed[i][4] = gesture

                #cv2.imshow('Hand ' + str(i+1), hand_frame)
                # ------HandGestures------#

            # ------LipDetection------#
            Talking = False
            if YAML_DATA['display_lip_detection']:
                gray_head_frame = cv2.cvtColor(head_frame, cv2.COLOR_BGR2GRAY)
                if len(zoomed[i][2]) != 0:
                    [distancevorige, breedtemondvorige, zerocount, talklist, Talking, counter] = zoomed[i][2]
                    distancevorige, breedtemondvorige, zerocount, talklist, Talking, counter = main_lip_detection2(head_frame, YAML_DATA, gray_head_frame, face_model, landmark_model, distancevorige, breedtemondvorige, zerocount, talklist, Talking, counter)
                else:
                    distancevorige, breedtemondvorige, zerocount, talklist, Talking, counter = main_lip_detection2(head_frame, YAML_DATA, gray_head_frame, face_model, landmark_model)
                zoomed[i][2] = [distancevorige, breedtemondvorige, zerocount, talklist, Talking, counter]
            # ------LipDetection------#
            # -----DisplayZoomed------#
            if YAML_DATA['display_face_detection_zoomed']:
                if (Talking or (not YAML_DATA['display_lip_detection'])) and (zoomed[i][4] or (not YAML_DATA['display_hand_gestures'])):
                    if len(zoomed[i][3]) == 0:
                        cv2.putText(head_frame, str(i + 1), (10, 380), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 1, cv2.LINE_AA)
                    else:
                        cv2.putText(head_frame, zoomed[i][3], (10, 380), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 1, cv2.LINE_AA)
                    show.append(head_frame)
                #elif not YAML_DATA['display_lip_detection'] and not YAML_DATA['display_hand_gestures']:
                #    show.append(head_frame)
            #organise(show)
            # -----DisplayZoomed------#
        i += 1
    #print('faces: ', faces)
    #print('zoomed: ', zoomed)
    while len(faces) <= i < len(zoomed):
        if zoomed[i][1] > 0:
            zoomed[i][1] -= 1
            if zoomed[i][1] >= YAML_DATA['tracking_treshhold_low']:
                x_start, x_eind, y_start, y_eind = make_frame(img, zoomed[i][0])
                head_frame = cv2.resize(img[y_start: y_eind, x_start: x_eind], (400, 400))
                if YAML_DATA['display_face_detection_zoomed']:
                    Talk = True
                    if YAML_DATA['display_lip_detection']:
                        if not zoomed[i][2][4]:
                            Talk = False
                    Gest = True
                    if YAML_DATA['display_hand_gestures']:
                        if not zoomed[i][4]:
                            Gest = False
                    if Talk and Gest:
                        if len(zoomed[i][3]) == 0:
                            cv2.putText(head_frame, str(i+1), (10, 380), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 1, cv2.LINE_AA)
                        else:
                            cv2.putText(head_frame, zoomed[i][3], (10, 380), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 1, cv2.LINE_AA)
                        show.append(head_frame)
                        #if len(zoomed[i][3]) == 0:
                            #cv2.imshow('Zoom in ' + str(i + 1), head_frame)
                        #else:
                            #cv2.imshow('Zoom in ' + zoomed[i][3], head_frame)
                    #elif not cv2.getWindowProperty('Zoom in ' + str(i + 1), cv2.WND_PROP_VISIBLE) < 1:
                    #    cv2.destroyWindow('Zoom in ' + str(i + 1))
            i += 1
        elif zoomed[i][1] == 0:
            #print('trying to remove')
            if cv2.getWindowProperty('Zoom in ' + str(i + 1), cv2.WND_PROP_VISIBLE) > 0:
                cv2.destroyWindow('Zoom in ' + str(i + 1))
            zoomed.remove(zoomed[i])


    # print('items in show', len(show))
    organise(show)

    # print('zoomed:', zoomed)
    # cv2.imshow('Live: ', img)


    # zoomed doorsturen zodat de face recognition dit kan gebruiken bij verder calculaties


    # print("zoomed in tracking", zoomed)
    # testconn2sender.send(zoomed)



            # # MOETEN WE NOG FIXEN !!!!!!
            # # ------LipDetection------#
            # # for i in zoomed
            # # (x, y, w, h) = faces[i]
            # zoomed_section = img[y:y + h, x:x + w]
            # # frame, YAML_DATA, distancevorige, gray_img, face_model, landmark_model, face_cascade
            # main_lip_detection(zoomed_section, YAML_DATA, distancevorige, gray_img, face_model, landmark_model, face_cascade)
            # # main_lip_detection(img, YAML_DATA, distancevorige, gray_img, face_model, landmark_model, face_cascade)
            # # ------LipDetection------#

