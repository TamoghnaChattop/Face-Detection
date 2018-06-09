# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 10:22:29 2018

@author: tchat
"""

import numpy as np
import cv2 as cv

face_cascade = cv.CascadeClassifier(r'C:\Users\tchat\.spyder-py3\Face Recognition\haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier(r'C:\Users\tchat\.spyder-py3\Face Recognition\haarcascade_eye.xml')

img = cv.imread(r'C:\Users\tchat\.spyder-py3\Face Recognition\testimg1.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        
cv.imshow('img',img)
cv.waitKey(0)
cv.destroyAllWindows()