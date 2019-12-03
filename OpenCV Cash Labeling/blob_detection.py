import cv2
import numpy as np

frameWidth = 640
frameHeight = 480

# cap = cv2.VideoCapture('train/sample7.mov')
cap = cv2.imread('extract/frame7.jpg', cv2.IMREAD_GRAYSCALE)

# cap.set(3, frameWidth)
# cap.set(4, frameHeight)

def empty():
	pass

cv2.namedWindow("Parameters")
# cv2.resizeWindow("Parameters",640, 240)

cv2.createTrackbar("minArea", "Parameters", 150, 244, empty)
cv2.createTrackbar("maxArea", "Parameters", 255, 255, empty)
# cv2.createTrackbar("minConvex", "Parameters", 150, 244, empty)
# cv2.createTrackbar("maxConvex", "Parameters", 255, 255, empty)
cv2.createTrackbar("minThreshold", "Parameters", 150, 244, empty)
cv2.createTrackbar("maxThreshold", "Parameters", 255, 255, empty)



while True:

	# Setup SimpleBlobDetector parameters.
	params = cv2.SimpleBlobDetector_Params()

	# Change thresholds
	params.minThreshold = cv2.getTrackbarPos("minThreshold", "Parameters")
	params.maxThreshold = cv2.getTrackbarPos("maxThreshold", "Parameters")
	# Filter by Area.
	params.filterByArea = 1
	params.minArea = cv2.getTrackbarPos("minArea", "Parameters")
	params.maxArea = cv2.getTrackbarPos("maxArea", "Parameters")


	if cv2.__version__.startswith('2.'):
		detector = cv2.SimpleBlobDetector()
	else:
		detector = cv2.SimpleBlobDetector_create(params)

	# Detect blobs.
	keypoints = detector.detect(cap)
	# print(keypoints)

	# Draw detected blobs as red circles.
	# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
	# the size of the circle corresponds to the size of blob

	# im_with_keypoints = cv2.drawKeypoints(cap, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

	# Show blobs
	cv2.imshow("Result", cap)

cv2.waitKey(0)
cv2.destroyAllWindows()