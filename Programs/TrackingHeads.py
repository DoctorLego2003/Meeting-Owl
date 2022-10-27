import cv2
import numpy
zoomed = []
face_cascade = cv2.CascadeClassifier(r'./xml/haarcascade_frontalface_default.xml')
profile_cascade = cv2.CascadeClassifier(r'./xml/haarcascade_profileface.xml')

def distance(point_one, point_two):
    if point_one == []:
        return float('inf')
    [y1, h0, x1, h0] = point_one
    [x2, y2, w1, h1] = point_two
    x1 = x1 + w2
    y1 = y1 + h2
    dist = ((x2 - x1)**2 + (y2 - y1)**2)**1/2
    #print(dist)
    return dist

'''
def closest(point, new_points):
    if len(new_points) == 0:
        return -1
    distmat = []
    for new_point in new_points:
        distmat.append(distance(point, new_point))
    index = 0
    minimum = distmat[0]
    for i in range(len(distmat)):
        if minimum > distmat[i]:
            minimum = distmat[i]
            index = i
    return index
'''
def detect_face_orientation(img):
    faces = face_cascade.detectMultiScale(gray_img, 1.25, 4)
    if len(faces) != 0:
        return faces
    else:
        faces = profile_cascade.detectMultiScale(gray_img, 1.3, 4)
        if len(faces) != 0:
            return faces
    gray_flipped = cv2.flip(gray_img, 1)
    faces = profile_cascade.detectMultiScale(gray_flipped, 1.3, 4)
    if len(faces) != 0:
        h, w = img.shape
        x = faces[0][0]
        new_x = h - x -1
        #print(w, x, new_x)
        faces[0][0] = new_x
    return faces

def sort_close(zoomed, faces):
    dist = []
    for i in range(len(zoomed)):
        dist.append([])
        for punt in faces:
            dist[i].append(distance(zoomed[i][1], punt))
    print('dist:', dist)
    min_ind = []
    for i in range(len(dist)):
        min_ind.append([])
        if len(dist[i]) != 0:
            mini = min(dist[i])
            if mini <= 300:
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
    #print(min_ind)
    i = 0
    while i < len(min_ind):
        if len(min_ind[i]) > 1:
            for point in min_ind[i]:
                if [point] in min_ind:
                    min_ind[i].remove(point)
        i += 1
    print(min_ind)
    return min_ind

cap = cv2.VideoCapture(0)

while True:

    ret, img = cap.read()
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #faces = face_cascade.detectMultiScale(gray_img, 1.25, 4)
    faces = detect_face_orientation(gray_img)

    print('faces before:',faces)
    #print('zoomed:',zoomed)
    min_ind = sort_close(zoomed, faces)
    #print('min_ind:', min_ind)

    #Trying to do Face Tracking
    missing_face = False
    new_faces = []
    for i in range(len(min_ind)):
        index = min_ind[i]
        #print('index:', index)
        if index == []:
            #print('iets')
            missing_face = True
        else:
            new_faces.append(faces[index[0]])

    if missing_face:
        if len(faces) >= len(zoomed):
            all_index = [x for x in range(len(faces))]
            missing_index = list(filter(lambda x:x not in min_ind, all_index))
            print('missing_index:', missing_index)
            face = faces[missing_index]
            new_faces.append(face)
            #print('new:', new_faces)
            missing_face = False
    if len(new_faces) != 0:
        if len(new_faces) > 1:
            #print(new_faces)
            faces = numpy.asarray(new_faces)
            #print(faces)
        else:
            faces = numpy.asarray(new_faces)

    if len(faces) > 0:
        #print(len(faces[0]))
        if len(faces) == 1 and len(faces[0]) != 4:
            faces = faces[0]
            #print(faces)

    #print('faces after:', faces)
    #print('zoomed:', zoomed)

    #Zooming in on the face
    for i in range(len(faces)):
        if len(faces) > len(zoomed):
            zoomed.append([[], [], ])
        #print('faces[i]:', faces[i])
        (x, y, w, h) = faces[i]
        print('faces[',i,']', faces)
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
        zoomed[i][0] = [y - h2, y + h + h2, x - w2, x + w + w2]
        #print('zoomed[i] new:', zoomed[i][0])

        #print(len(gray_img))
        #print(len(gray_img[0]))
        cv2.imshow('Zoom in ' + str(i + 1), img[zoomed[i][0][0]: zoomed[i][0][1], zoomed[i][0][2]: zoomed[i][0][3]])
        cv2.resizeWindow('Zoom in ' + str(i + 1), 325, 325)

    print('zoomed:', zoomed)
    #print('prev:',prev)
    removed = []
    for i in range(len(zoomed)):
        if zoomed[i][0] == zoomed[i][1]:
            cv2.destroyWindow('Zoom in ' + str(i+1))
            removed.append(zoomed[i])
        else:
            zoomed[i][1] = zoomed[i][0]
    for x in removed:
        zoomed.remove(x)

    '''
    if len(faces) != len(prev) and len(prev) != 0:
        cv2.destroyWindow('Zoom in ' + str(len(zoomed)))
        zoomed.remove(zoomed[-1])
    '''

    cv2.imshow('Face Recognition', img)
    #if start and zoomed.all() is not None:
    #    cv2.imshow('Zoom in', zoomed)


    k = cv2.waitKey(30) & 0xff
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
