# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 23:54:13 2018

@author: impawan


"""

import cv2
import numpy as np

img = cv2.imread('bookpage.jpg')
#normal threshold also called fixed threshold that is applied on BGR image
retval,thershold = cv2.threshold(img,12,255,cv2.THRESH_BINARY)

#threshold of image after changibg it to greyscale
geryscaled = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

retval2,thershold2 = cv2.threshold(geryscaled,12,255,cv2.THRESH_BINARY)


#using adapative threshold
gaus = cv2.adaptiveThreshold(geryscaled,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,115,1)
cv2.imshow('orignal',img)

cv2.imshow('thres',thershold) 
 
cv2.imshow('thres2',thershold2)
cv2.imshow('gaus',gaus)




cv2.waitKey(0)

cv2.destroyAllWindows()