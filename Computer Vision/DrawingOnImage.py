# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 12:44:12 2018

@author: impawan


"""


import cv2
import numpy as np


#opencv has drawing libs

img = cv2.imread('mchinevision.jpg',cv2.IMREAD_COLOR)

cv2.line(img,(0,0),(150,150),(255,255,255),15)  #15 sets the width of line

cv2.rectangle(img,(0,0),(150,150),(0,0,255),15)
cv2.circle(img,(100,63),23,(0,0,255),-1) #23 is radius and -1 to fill the radius with red color

#we can plot polygons also for this we need to specify an array where we wil have all the points stored 
pts = np.array([[10,20],[20,30],[70,20],[50,10]],np.int32) #to plot plogons will e required poinst that will be connetced
cv2.polylines(img,[pts],True,(0,255,255),3)

#writing text on the image, we need to declare fonts

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,'Pawan ',(200,200),font,2,(200,200,200),2,cv2.LINE_AA)


cv2.imshow('image,',img)

cv2.waitKey(0)
cv2.destroyAllWindows()