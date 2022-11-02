import cv2
import face_recognition
import time

from git.Programs.functions.simple_facerec import SimpleFacerec


ctime = 0
ptime = 0

# Load the cascade
face_cascade = cv2.CascadeClassifier('xml/haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# To capture video from webcam.
cap = cv2.VideoCapture(0)
# To use a video file as input
# cap = cv2.VideoCapture('filename.mp4')
# cap = cv2.VideoCapture('vids/WIN_20221013_15_10_37_Pro.mp4')

# # Load Camera
# cap = cv2.VideoCapture(2)
# Encode faces from a folder
sfr = SimpleFacerec()
sfr.load_encoding_images("images/")
names = []




while True:
    ret, frame = cap.read()


    # Detect Faces
    face_locations, face_names = sfr.detect_known_faces(frame)  # HOERE TRAAAAAGGGGG


    # print(face_locations, face_names)

    for face_loc, name in zip(face_locations, face_names):
        print(name)

        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
        # print(face_loc)

        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

    key = cv2.waitKey(1)
    if key == 27:
        break

    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime

    cv2.putText(frame, f'FPS:{str(int(fps))}', (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)

    cv2.imshow("Frame", frame)


def export_names(ret, frame):
    sfr = SimpleFacerec()
    sfr.load_encoding_images("images/")
    face_locations, face_names = sfr.detect_known_faces(frame)
    return face_names
