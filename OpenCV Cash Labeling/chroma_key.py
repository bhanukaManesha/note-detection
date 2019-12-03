import cv2
import numpy as np

vidcap = cv2.VideoCapture('train/sample6.mov')

success,image = vidcap.read()



count = 0
while success:
    # cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file      
    success,frame = vidcap.read()


    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) 
    # cv2.imshow("img", gray)

    bi = cv2.bilateralFilter(gray, 5, 75, 75)
    # cv2.imshow('bi',bi)

    dst = cv2.cornerHarris(bi, 2, 3, 0.04)

    mask = np.zeros_like(gray)

    mask[dst>0.01*dst.max()] = 255
    # cv2.imshow('mask', mask)


    # # cv2.rectangle(image, (find_x_y[1][-1],find_x_y[0][0]),(find_x_y[1][0],find_x_y[0][-1]),(0, 255, 0), 2)
    cv2.imwrite("frame%d.jpg" % count, mask)

    print('Read a new frame: ', success)
    count += 1