import copy

import cv2
import numpy
# prev = []
# zoomed = []
# face_cascade = cv2.CascadeClassifier(r'./xml/haarcascade_frontalface_default.xml')
# profile_cascade = cv2.CascadeClassifier(r'./xml/haarcascade_profileface.xml')

from .LipDetection import *


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
    left_profile = profile_cascade.detectMultiScale(gray_img, 1.3, 2)
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
    right_profile = profile_cascade.detectMultiScale(gray_flipped, 1.3, 2)

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
    for face in faces:
        (x, y, w, h) = face
        #assert x >= 0
        #assert y >= 0
        #assert w >= 0
        #assert h >= 0
    check_for_double_faces(faces)
    check_for_empty_faces(faces)
    return faces

def distance(point_one, point_two):
    #print((point_one, point_two))
    if point_one == []:
        return float('inf')
    [x1, y1, w0, h0] = point_one
    [x2, y2, w1, h1] = point_two
    dist = ((x2 - x1)**2 + (y2 - y1)**2)**1/2
    #print(dist)
    return dist

def track(faces, zoomed):
    if len(zoomed) == 0:
        return faces
    print('zoomed: ', zoomed)
    dist = []
    for i in range(len(zoomed)):
        dist.append([])
        for punt in faces:
            dist[i].append(distance(zoomed[i][0], punt))
    #print('dist:', dist)
    #calculating lowest distance per point
    min_ind = []
    for i in range(len(dist)):
        min_ind.append([])
        if len(dist[i]) != 0:
            mini = min(dist[i])
            if mini <= 100:
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
    print('min_ind:', min_ind)
    i = 0
    while i < len(min_ind):
        if len(min_ind[i]) > 1:
            for point in min_ind[i]:
                if [point] in min_ind:
                    min_ind[i].remove(point)

        i += 1
    #returning a reorganised faces
    print('faces: ', faces)
    new_faces = []
    missing_index = []
    for i in range(len(min_ind)):
        index = min_ind[i]
        print('index:', index)
        if index != []:
            new_faces.append(faces[index[0]])
        else:
            missing_index.append(i)
    print('missing_index1:', missing_index)
    if (len(new_faces) > 0) and (len(missing_index) > 0):
        print('missing_index:', missing_index)
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
        print('missing_index:', missing_index)
        for i in range(len(missing_index)):
            new_faces.append(faces[missing_index[i]])
        #print('new:', new_faces)
    #print('faces: ', faces)
    #print('new_faces: ', new_faces)
    print('faces1:', faces)
    check_for_double_faces(new_faces)
    print('faces2:', faces)
    check_for_empty_faces(new_faces)
    print('faces3:', faces)

    if len(zoomed) == len(new_faces):
        c = 0.8
        for i in range(len(zoomed)):
            for j in range(4):
                new_faces[i][j] = int(c * new_faces[i][j] + (1 - c) * zoomed[i][0][j])
    return new_faces

# try:
#     cap = cv2.VideoCapture(1)
#     ret, img = cap.read()
#     assert len(img) > 0
# except:
#     cap = cv2.VideoCapture(0)

def check_for_doubles(zoomed):
    for i in range(len(zoomed)):
        j = i + 1
        while j < len(zoomed):
            if zoomed[i][0] == zoomed[j][0] and zoomed[i][1] == zoomed[j][1]:
                zoomed.remove(zoomed[j])
                #print('new zoom: ', zoomed)
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

# while True:
def main_tracking(img, YAML_DATA, zoomed, gray_img, face_cascade, profile_cascade, distancevorige, face_model, landmark_model):

#     ret, img = cap.read()
#     gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     #faces = face_cascade.detectMultiScale(gray_img, 1.25, 4)
    faces = detect_face_orientation(gray_img, face_cascade, profile_cascade)
    # print('faces:', faces, len(faces))
    faces = track(faces, zoomed)
    check_for_doubles(zoomed)
    check_for_empty(zoomed)
    for i in range(len(faces)):
        if len(faces) > len(zoomed):
            zoomed.append([[], 0, str()])
        (x, y, w, h) = faces[i]

        # -----SHOW RECTANGLE-----#
        if YAML_DATA['display_face_detection'] == True:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
        # -----SHOW RECTANGLE-----#


        # ------SHOW ZOOMED------#
        if YAML_DATA['display_face_detection_zoomed'] == True:
            rec_gray = gray_img[y:y + h, x:x + w]
            rec_color = img[y:y + h, x:x + w]
            htot = 3 * h // 2
            wtot = 3 * w // 2
            h2 = (htot - h) // 2
            w2 = (wtot - w) // 2
            if y < h2:
                h2 = y
            if x + w2 + w > len(gray_img[0]):
                w2 = len(gray_img[0]) - w - x
            if x < w2:
                w2 = x
            if y + h + h2 > len(gray_img):
                h2 = len(gray_img) - y - h

            if zoomed[i][0] != faces[i] or len(zoomed[i][0]) == 0:
                zoomed[i][0] = faces[i]
                if zoomed[i][1] < YAML_DATA['tracking_treshhold_high']:
                    zoomed[i][1] += 1
            elif zoomed[i][1] > 0:
                zoomed[i][1] -= 1

            if (zoomed[i][1] >= YAML_DATA['tracking_treshhold_low']) and (zoomed[i][1] <= YAML_DATA['tracking_treshhold_high']):
                head_frame = img[y - h2: y + h2 + h, x - w2: x + w2 + w]
                head_frame = cv2.resize(head_frame, (400, 400))
                if YAML_DATA['display_lip_detection']:
                    main_lip_detection(head_frame, YAML_DATA, distancevorige, head_frame, face_model, landmark_model, face_cascade)
                cv2.imshow('Zoom in ' + str(i + 1), head_frame)

            if zoomed[i][1] == 0:
                if cv2.getWindowProperty('Zoom in ' + str(i + 1), cv2.WND_PROP_VISIBLE) > 0:
                    cv2.destroyWindow('Zoom in ' + str(i + 1))
                zoomed.remove(zoomed[i])


    for i in range(len(faces), len(zoomed)):
        if i >= len(zoomed):
            break
        print('i:', i)
        if zoomed[i][1] > 0:
            zoomed[i][1] -= 1
            if zoomed[i][1] >= YAML_DATA['tracking_treshhold_low']:
                [x, y, w, h] = zoomed[i][0]
                htot = 3 * h // 2
                wtot = 3 * w // 2
                h2 = (htot - h) // 2
                w2 = (wtot - w) // 2
                if y < h2:
                    h2 = y
                if x + w2 + w > len(gray_img[0]):
                    w2 = len(gray_img[0]) - w - x
                if x < w2:
                    w2 = x
                if y + h + h2 > len(gray_img):
                    h2 = len(gray_img) - y - h
                head_frame = cv2.resize(img[y - h2: y + h2 + h, x - w2: x + w2 + w], (400, 400))
                cv2.imshow('Zoom in ' + str(i + 1), head_frame)
        print('zoomed:', zoomed)
        if zoomed[i][1] == 0:
            if cv2.getWindowProperty('Zoom in ' + str(i + 1), cv2.WND_PROP_VISIBLE) > 0:
                cv2.destroyWindow('Zoom in ' + str(i + 1))
            zoomed.remove(zoomed[i])

        # cv2.imshow('Live: ', img)




            # # MOETEN WE NOG FIXEN !!!!!!
            # # ------LipDetection------#
            # # for i in zoomed
            # # (x, y, w, h) = faces[i]
            # zoomed_section = img[y:y + h, x:x + w]
            # # frame, YAML_DATA, distancevorige, gray_img, face_model, landmark_model, face_cascade
            # main_lip_detection(zoomed_section, YAML_DATA, distancevorige, gray_img, face_model, landmark_model, face_cascade)
            # # main_lip_detection(img, YAML_DATA, distancevorige, gray_img, face_model, landmark_model, face_cascade)
            # # ------LipDetection------#

