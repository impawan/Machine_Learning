# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 22:46:48 2018

@author: ACER
"""

import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('G:/Machine Learning/Computer Vision/classfier.xml')
eye_cascade = cv2.CascadeClassifier('G:/Machine Learning/Computer Vision/haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

while True:
    ret,frame = cap.read()
    grey_scale = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grey_scale,1.3,5)
    for (faceX,faceY,faceW,faceH) in faces:
        cv2.rectangle(frame,(faceX,faceY),(faceX+faceW,faceY+faceH),(255,0,0),1)
        roi_grey = grey_scale[faceY:faceY+faceH,faceX:faceX+faceX+faceW]
        roi_color = frame[faceY:faceY+faceH,faceX:faceX+faceX+faceW]
        eyes = eye_cascade.detectMultiScale(roi_grey)
        for (x,y,w,h) in eyes:
            cv2.rectangle(roi_color,(x,y),(x+w,y+h),(0,0,255),1)
            
    cv2.imshow('video',frame)   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()    