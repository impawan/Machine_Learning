# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 11:50:10 2018

@author: ACER
"""

import cv2
import numpy as np

cap =cv2.VideoCapture(0)

while True:
    _,frame = cap.read()
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    lower_black = np.array([130,50,0])
    upper_black = np.array([180,245,235])
    
    print (frame[100,100],frame[100,629])
    frame [100,100]=[0,0,255]
    
    mask = cv2.inRange(hsv,lower_black,upper_black)
    res = cv2.bitwise_and(frame,frame, mask = mask)
    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27: 
        break 
    
cv2.destroyAllWindows()
cap.release()   