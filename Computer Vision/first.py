# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 10:37:00 2018

@author: impawan
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('mchinevision.jpg',cv2.IMREAD_GRAYSCALE)


#displayig image using cv2
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


#Diplaying image using matplotlib
#
#plt.imshow(img,cmap='gray',interpolation='bicubic')
#plt.plot([50,100],[80,100],'c',linewidth=5)
#plt.show()


#saving the image 

cv2.imwrite('cv2Image.png',img)