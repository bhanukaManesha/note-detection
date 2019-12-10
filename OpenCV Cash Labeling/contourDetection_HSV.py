import cv2
import numpy as np
from matplotlib import pyplot as plt

frameWidth = 320
frameHeight = 240

# Script parameters

testingMode = True
collectData = False
currency = 1
side = "front"

# cap = cv2.VideoCapture('dataset/videos/' + str(currency) + '_'+ side + '.mov')
cap = cv2.VideoCapture('experiment/greenscreen_50.mov')

cap.set(3, frameWidth)
cap.set(4, frameHeight)

def empty():
	pass

cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters",640, 240)
cv2.createTrackbar("Threshold1", "Parameters", 26, 244, empty)
cv2.createTrackbar("Threshold2", "Parameters", 90, 255, empty)
cv2.createTrackbar("AreaMin", "Parameters", 150000, 550000, empty)
# cv2.createTrackbar("AreaMax", "Parameters", 500000, 550000, empty)

def getContours(img,imgContour, count):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        areaMin = cv2.getTrackbarPos("AreaMin", "Parameters")
        # areaMax = cv2.getTrackbarPos("AreaMax", "Parameters")

        # if the area is more than the minimium
        if area > areaMin:

            # Draw the contours on the image
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 255), 7)

            # Apporximate the contours
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            # Getting the bounding box
            x , y , w, h = cv2.boundingRect(approx)

            # Extract only if the image size is more than 128
            
            cv2.rectangle(imgContour, (x , y ), (x + w , y + h ), (0, 255, 0), 5)

            # Resize
            border_v = 0
            border_h = 0
            IMG_COL = 128
            IMG_ROW = 128

            # imgFinal[:,:,0] = imgCC[:,:,0] * green
            # imgFinal[:,:,1] = imgCC[:,:,1] * green
            # imgFinal[:,:,2] = imgCC[:,:,2] * green
            imgFinal = imgCC * imgRGBGray
            extractImg = imgFinal[y:y+h,x:x+w]

            # if (IMG_COL/IMG_ROW) >= (extractImg.shape[0]/extractImg.shape[1]):
            #     border_v = int((((IMG_COL/IMG_ROW)*extractImg.shape[1])-extractImg.shape[0])/2)
            # else:
            #     border_h = int((((IMG_ROW/IMG_COL)*extractImg.shape[0])-extractImg.shape[1])/2)
            # extractImg = cv2.copyMakeBorder(extractImg, border_v, border_v, border_h, border_h, cv2.BORDER_CONSTANT, 0)
            # finalImg = cv2.resize(extractImg, (1920, 1080))

            if collectData :
                cv2.imwrite("dataset/image/RM"+ str(currency)  + "/rm"+ str(currency) + "_"+ side + "_" +"%d.jpg" % count, finalImg)
                print(str(count) + ": " +  str(w) + " " + str(h) + " " + str(y+h) + " " + str(x+w))

            if testingMode :                
                cv2.imwrite("testing/%d.jpg" % count, extractImg)

            cv2.putText(imgContour, "Points: " + str(len(approx)), (x + w + 20, y + 20), cv2.FONT_HERSHEY_COMPLEX, .7,
                        (0, 255, 0), 2)
            cv2.putText(imgContour, "Area: " + str(int(area)), (x + w + 20, y + 45), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                        (0, 255, 0), 2)

def stackImages(scale,imgArray):
    '''
    Method to stack all the images and easily display them
    '''
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

# Initialize variables
count = 0

while True:

    # Read each frame
    sucess, img = cap.read()
    imgFinal = img.copy()
    imgCC = img.copy()

    # Convert BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    ## mask of green (36,25,25) ~ (86, 255,255)
    # mask = cv2.inRange(hsv, (36, 25, 25), (86, 255,255))
    mask = cv2.inRange(hsv, (36, 25, 25), (70, 255,255))

    ## slice the green
    imask = mask == 0
    green = np.zeros_like(img, np.uint8)
    green[imask] = img[imask]

    img = cv2.cvtColor(green, cv2.COLOR_HSV2BGR)

    imgRGBGray = green != 0


    

    

    # finalImg[:, :, 0] = finalImg[:,:,0] * alpha
    # finalImg[:, :, 1] = finalImg[:,:,1] * alpha
    # finalImg[:, :, 2] = finalImg[:,:,2] * alpha

    # finalImg = cv2.resize(finalImg, (IMG_ROW, IMG_COL))

    # #  Display the images
    # imgStack = stackImages(0.5, ([img, green], [bgrimg, bgrimg]))
    # cv2.imshow("Result", imgStack)

    # Take copy of the frame
    imgContour = img.copy()

    # Add gaussian blur
    imgBlur = cv2.GaussianBlur(imgContour, (9,9), 1)

    # Convert the image to gray
    imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)

    # Get the values from the tracker bar
    threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
    threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")

    # Get the canny edge image
    imgCanny = cv2.Canny(imgGray, threshold1, threshold2)

    # Create a kernal of ones
    kernal = np.ones((7,7))

    # Dialate the image to expand the enhance the edges
    imgDil = cv2.dilate(imgCanny, kernal, iterations=1)

    # Get the contours
    getContours(imgDil, imgContour, count)

    # Display the images
    imgStack = stackImages(0.5, ([img, imgCanny], [imgDil, imgContour]))
    cv2.imshow("Result", imgStack)

    # Increase the count
    count += 1

    # Exit from the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

