# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 08:49:04 2018

@author: impawan
"""



import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('D:/Users/paprasad/python/classfier.xml')


image = cv2.imread('D:/Users/paprasad/python/download.jpg')
grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


faces = face_cascade.detectMultiScale(grayImage)
print (type(faces))
print (faces)

print ("Number of faces detected: " + str(faces.shape[0]))


for (x,y,w,h) in faces:
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),1)
 
cv2.rectangle(image, ((0,image.shape[0] -25)),(270, image.shape[0]), (255,255,255), -1)
cv2.putText(image, "Number of faces detected: " + str(faces.shape[0]), (0,image.shape[0] -10), cv2.FONT_HERSHEY_TRIPLEX, 0.5,  (0,0,0), 1)
 
cv2.imshow('Image with faces',image)
cv2.waitKey(0)
cv2.destroyAllWindows()