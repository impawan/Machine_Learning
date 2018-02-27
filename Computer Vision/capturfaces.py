# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 22:18:25 2018

@author: impawan
"""

import cv2
import numpy as np
import webbrowser


face_cascade = cv2.CascadeClassifier("D:\Machine Learning\Computer Vision\classfier.xml")

cap = cv2.VideoCapture(0)

while True :
    ret , frame = cap.read()
    faceCount = 0
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    
    
    for (x,y,w,h) in faces:
        faceCount=int(faces.shape[0])
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)
    cv2.imshow('Face Detection',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if faceCount==2:
        webbrowser.open('file:///D:/Machine%20Learning/Computer%20Vision/Machine_Learning/Computer%20Vision/stop_looking_at_my_screen.jpg', new=2)
        break
cap.release()
cv2.destroyAllWindows()   
        
        
 
    

    