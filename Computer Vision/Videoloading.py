# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 11:00:02 2018

@author: impawan
"""


import cv2
import numpy as np

vid = cv2.VideoCapture(0) #0 means fisrt webcam if you have another cam then it will be 1 and so on.
fourcc = cv2.VideoWriter_fourcc(*'XVID') #define the codec to write the video file
out = cv2.VideoWriter('outvideo.avi',fourcc,20.0,(640,480)) #filname and codec 

while True:
    ret, frame = vid.read()
    # we can display more than oen frame and we can also have grey version of fram
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    out.write(frame) #wrting the frame into file
    cv2.imshow('frame',frame)  
    cv2.imshow('geryFrame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
vid.release()
out.release()
cv2.destroyAllWindows()    