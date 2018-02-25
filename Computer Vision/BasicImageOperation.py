# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 13:12:05 2018

@author: impawan
"""

import cv2
import numpy as np
img = cv2.imread('mchinevision.jpg',cv2.IMREAD_COLOR)

#to access any pixel
#
#px =img[55,55]
#print (px)


#to modify the pixel

img[55,55] = [255,255,255]
#roi region of image trucnating the image an filling it with white space
#img[325:800,185:600]=[255,255,255]
##
my_face = img[325:800,185:600]
img[0:475,0:415] = my_face
#cv2.imshow('face',my_face)
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()