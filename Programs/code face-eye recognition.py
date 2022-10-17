import cv2

face_cascade = cv2.CascadeClassifier(r'./xml/haarcascade_frontalface_default.xml')
profile_cascade = cv2.CascadeClassifier(r'./xml/haarcascade_profileface.xml')

def detect_face_orientation(img):
    faces = []
    front_faces = face_cascade.detectMultiScale(gray_img, 1.25, 4)
    left_profile = profile_cascade.detectMultiScale(gray_img, 1.3, 2)
    gray_flipped = cv2.flip(gray_img, 1)
    right_profile = profile_cascade.detectMultiScale(gray_flipped, 1.3, 2)
    if len(front_faces) != 0:
        for face in front_faces:
            faces.append(face)
    if len(left_profile) != 0:
        for face in left_profile:
            faces.append(face)

    if len(right_profile) != 0:
        for face in right_profile:
            h, w = img.shape
            x = face[0]
            new_x = (w/2) - x - 1
            face[0] = new_x
            faces.append(face)
    return faces

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detect_face_orientation(gray_img)


    """
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
        rec_gray = gray_img[y:y + h, x:x + w]
        rec_color = img[y:y + h, x:x + w]

        zoomed = img[y:y + h, x:x + w]
        start = True
        #eyes = eye_cascade.detectMultiScale(rec_gray)

        #for (a, b, c, d) in eyes:
        #    cv2.rectangle(rec_color, (a, b), (a+c, b+d), (0, 127, 255), 2)
    """
    zoomed = []
    for i in range(len(faces)):
        (x, y, w, h) = faces[i]
        zoomed.append([])
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
        rec_gray = gray_img[y:y + h, x:x + w]
        rec_color = img[y:y + h, x:x + w]
        h2 = 75
        w2 = 75
        if y < h2:
            h2 = y
        if x + w2 + w > len(gray_img[0]):
            w2 = len(gray_img[0]) - w - x
        if x < w2:
            w2 = x
        if y + h + h2 > len(gray_img):
            h2 = len(gray_img) - y - h
        zoomed[i] = img[y-h2:y + h +h2, x -w2:x + w +w2]
        #print(len(gray_img))
        #print(len(gray_img[0]))
        cv2.imshow('Zoom in ' + str(i + 1), zoomed[i])
        cv2.resizeWindow('Zoom in ' + str(i + 1), 325, 325)

    cv2.imshow('Face Recognition', img)
    #if start and zoomed.all() is not None:
    #    cv2.imshow('Zoom in', zoomed)

    k = cv2.waitKey(30) & 0xff
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()