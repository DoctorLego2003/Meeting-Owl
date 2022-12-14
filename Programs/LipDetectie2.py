import numpy as np
import cv2
import dlib
import time
from scipy.spatial import distance as dist
from imutils import face_utils



def cal_yawn(shape, distancevorige, breedtemondvorige):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))

    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)
    breedteface = (((shape[0][0]-shape[16][0])**2)+((shape[0][1]-shape[16][1])**2))

    distancenu = dist.euclidean(top_mean, low_mean) ** 2

    breedtemondnu = (((shape[48][0]-shape[54][0])**2)+((shape[48][1]-shape[54][1])**2))

    #print(breedtemondnu)
    return distancenu, distancevorige, breedtemondnu, breedtemondvorige, breedteface


#cam = cv2.VideoCapture('C:/Users/arnel/OneDrive/Documents/Burgi/Semester 3/Peno/Tests/Spraak/Test 2 speech.mp4')
# cam = cv2.VideoCapture(0)
# -------Models---------#
# face_model = dlib.get_frontal_face_detector()
# landmark_model = dlib.shape_predictor('./dat/shape_predictor_68_face_landmarks.dat')
# --------Variables-------#
# yawn_thresh = 35

# ptime = 0
# while True:
def main_lip_detection2(frame, YAML_DATA, gray_img, face_model, landmark_model, distancevorige=0, breedtemondvorige=1, zerocount=0, talklist=0, Talking=False, counter = 0):
    # distancevorige = 0
    # breedtemondvorige = 1
    # zerocount = 0
    # talklist = 0
    # Talking = False

    #print("running")

    # yawn_tresh = YAML_DATA['yawn_threshold']

    # ret, img = cam.read()
    # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #
    #
    # suc, frame = cam.read()
    #
    # if not suc:
    #     break

    # # ---------FPS------------#
    # ctime = time.time()
    # fps = int(1 / (ctime - ptime))
    # ptime = ctime
    # cv2.putText(frame, f'FPS:{fps}', (frame.shape[1] - 120, frame.shape[0] - 20), cv2.FONT_HERSHEY_PLAIN, 2,
    #            (0, 200, 0), 3)

    # ------Detecting face------#
    # img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_gray = gray_img
    faces = face_model(img_gray)
    '''
    if len(faces) == 0:
        counter += 1
        if counter > 20:
            Talking = False
            counter = 0
        return distancevorige, breedtemondvorige, zerocount, talklist, Talking, counter
    '''
    for face in faces:


        # ----------Detect Landmarks-----------#
        shapes = landmark_model(img_gray, face)
        shape = face_utils.shape_to_np(shapes)
        #print(shapes)
        #print(shape)
        # -------Detecting/Marking the lower and upper lip--------#
        #lip = shape[48:60]
        #cv2.drawContours(frame, [lip], -1, (0, 165, 255), thickness=3)


        # -------Calculating the lip distance-----#
        lip_dist, distancevorige, breedtemond, breedtemondvorige, breedteface = (cal_yawn(shape, distancevorige, breedtemondvorige))
        #breedteface = int(breedteface)
        #lip_dist = int(lip_dist)
        #breedtemond = int(breedtemond)
        #distancevorige = int(distancevorige)
        #breedtemondvorige = int(breedtemondvorige)
        verschilmond = abs(breedtemond - breedtemondvorige)
        verschillip = abs(lip_dist - distancevorige)
        #print(breedtemond)
        #if verschilmond <= 1:
        #    verschilmond = 1
        #print(verschilmond)
        #print("verschillip",verschillip)
        #print("verschilmond", verschilmond)
        #if breedteface == 0:
        #    breedteface == 1
        relatief_verschil = verschillip * verschilmond / breedteface * 100
        #print(relatief_verschil)
        #print(relatief_verschil)


        # if relatief_verschil >=10:

        if relatief_verschil >= YAML_DATA['relatief_verschil_waarde']:
            talklist += 1
            #print(talklist)
            #if zerocount >= 10:
            #    talklist= 0
            if talklist >= 8:
                Talking = True
                zerocount = 0
                #print("Talking")
        else:
            zerocount += 1
            # if zerocount >= 10:
            if zerocount >= YAML_DATA['zerocount_buffer_count']:
                zerocount = 0
                talklist = 0
                Talking = False
                #print("Not Talking")
        if distancevorige == lip_dist and breedtemondvorige == breedtemond:
            Talking = False

        if YAML_DATA['display_lip_output']:
            if Talking:
                cv2.putText(frame, "Talking", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0), 2)
                #print("Talking")
            else:
                cv2.putText(frame, "Not Talking", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 200), 2)
            #print("Not talking")

        distancevorige = lip_dist
        breedtemondvorige = breedtemond

    return distancevorige, breedtemondvorige, zerocount, talklist, Talking, counter

    # cv2.imshow('Webcam', frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# cam.release()
# cv2.destroyAllWindows()