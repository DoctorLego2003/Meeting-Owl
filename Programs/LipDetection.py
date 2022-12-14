
import numpy as np
import cv2
import dlib
import time
from scipy.spatial import distance as dist
from imutils import face_utils

# distancevorige = 0

# face_model = dlib.get_frontal_face_detector()
# landmark_model = dlib.shape_predictor('./dat/shape_predictor_68_face_landmarks.dat')

# face_cascade = cv2.CascadeClassifier('./xml/haarcascade_frontalface_default.xml')


def cal_yawn(shape, distancevorige):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))

    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    distance = dist.euclidean(top_mean, low_mean)
    distancenu = distance

    return distancenu, distancevorige


# cam = cv2.VideoCapture(0)

# -------Models---------#
# face_model = dlib.get_frontal_face_detector()
# landmark_model = dlib.shape_predictor('./dat/shape_predictor_68_face_landmarks.dat')
# print(face_model)
# --------Variables-------#
# yawn_thresh = 35
# ptime = 0


# while True:
def main_lip_detection(frame, YAML_DATA, distancevorige, gray_img, face_model, landmark_model, face_cascade):
    yawn_thresh = YAML_DATA['yawn_threshold']


    # ret, img = cam.read()
    # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gezicht = face_cascade.detectMultiScale(gray_img, 1.25, 4)

    # cam = frame
    # suc, frame = cam.read()

    # if suc:

    # ---------FPS------------#
    # ctime = time.time()
    # fps = int(1 / (ctime - ptime))
    # ptime = ctime
    # cv2.putText(frame, f'FPS:{fps}', (frame.shape[1] - 120, frame.shape[0] - 20), cv2.FONT_HERSHEY_PLAIN, 2,
    #             (0, 200, 0), 3)


    # ------Detecting face------#
    # img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_gray = gray_img
    faces = face_model(img_gray)
    for face in faces:

        # #------Uncomment the following lines if you also want to detect the face ----------#
        # x1 = face.left()
        # y1 = face.top()
        # x2 = face.right()
        # y2 = face.bottom()
        # # print(face.top())
        # cv2.rectangle(frame,(x1,y1),(x2,y2),(200,0,00),2)

        # ----------Detect Landmarks-----------#
        shapes = landmark_model(img_gray, face)
        shape = face_utils.shape_to_np(shapes)

        # -------Detecting/Marking the lower and upper lip--------#
        lip = shape[48:60]
        cv2.drawContours(frame, [lip], -1, (0, 165, 255), thickness=3)

        # -------Calculating the lip distance-----#

        lip_dist, distancevorige = (cal_yawn(shape, distancevorige))
        lip_dist = int(lip_dist)
        distancevorige = int(distancevorige)
        verschil = abs(lip_dist - distancevorige)
        for (x, y, w, h) in gezicht:
            relatief_verschil = 100*verschil/w
            # print(relatief_verschil)
            if relatief_verschil >= 5:
                cv2.putText(frame, "Talking", (frame.shape[1] // 2 - 170, frame.shape[0] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 200), 2)
        #cv2.putText(frame, lip_dist, (frame.shape[1] // 2 - 170, frame.shape[0] // 2),
        #            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 200), 2)


        distancevorige = lip_dist
        """
        if lip_dist > yawn_thresh:
            cv2.putText(frame, f'User Yawning!', (frame.shape[1] // 2 - 170, frame.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 200), 2)
        """
    # cv2.imshow('Webcam', frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# cam.release()
# cv2.destroyAllWindows()