import cv2

# Load the cascade
face_cascade = cv2.CascadeClassifier('../xml/haarcascade_frontalface_default.xml')
# Read the input image
img = cv2.imread('test.jpg')
# Convert into grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Detect faces
faces = face_cascade.detectMultiScale(gray, 1.1, 4)
# Draw rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
# Display the output
cv2.imshow('img', img)
#gray[0,0]=5
#gray[1,1]=5
#gray[2,2]=5
#gray[3,3]=5
#gray[4,4]=5
#gray[5,5]=5
#gray[6,6]=5
#gray[7,7]=5
#cv2.imshow('gray', gray)
cv2.waitKey()