import cv2
import numpy as np
from matplotlib import pyplot as plt

im = cv2.imread('Canopic-Jar.jpg')
imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
imgray = cv2.GaussianBlur(imgray,(5,5),0)
#ret, thresh = cv2.threshold(imgray,100,255,0)
thresh = cv2.adaptiveThreshold(imgray,255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)


kernel1 = np.ones((3,3),np.uint8)
kernel2 = np.ones((5,5),np.uint8)
erosion = cv2.erode(im,kernel2,iterations = 1)
dilation = cv2.dilate(im,kernel2,iterations = 1)
opening = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel2)
closing = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel2)


plt.figure(figsize=(24,8))
plt.subplot(1,3,1),plt.imshow(thresh,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,2),plt.imshow(closing,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,3),plt.imshow(erosion,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.show()