# Imports
import numpy as np
import cv2 as cv

# Get image analysis cascades
face_cascade = cv.CascadeClassifier('opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('opencv\\build\\etc\\haarcascades\\haarcascade_eye.xml')

# Load image
img = cv.imread('test.jpg')

# Calculate image scale
scale_percent = 30 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)

# Resize the image
img = cv.resize(img, dim, interpolation = cv.INTER_AREA)

# Turn to gray scale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Get faces
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

# Loop through faces
for (x,y,w,h) in faces:
    cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

# Show results
cv.imshow('img',img)
cv.waitKey(0)
cv.destroyAllWindows()
