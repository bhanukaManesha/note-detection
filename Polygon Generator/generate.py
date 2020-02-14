import cv2
import numpy as np

img = cv2.imread('data/1.png', 0)
img = cv2.medianBlur(img, 5)
thresh = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)

cv2.imshow('Image', thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()


contours, hierarchy =   cv2.findContours(thresh.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
epsilon = 0.1*cv.arcLength(cnt,True)
approx = cv.approxPolyDP(cnt,epsilon,True)

cv2.drawContours(img, approx, -1, (0,255,0), 3)
cv2.imshow('Contours', img)
cv2.waitKey(0)
cv2.destroyAllWindows()