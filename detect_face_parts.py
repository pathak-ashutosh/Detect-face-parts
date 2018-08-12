from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2

# constructing the argument parser and parsing the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required = True,
	help = "path to facial landmark predictor")
ap.add_argument("-i", "--image", required = True,
	help = "path to input image")
args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create the
# facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# load the input image, resize it, and convert it to grayscale
image = cv2.imread(args["image"])
image = imutils.resize(image, width = 500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect faces in the grayscale image
rects = detector(gray, 1)



# Now that we have detected faces in the image, we can loop over each of
# the face ROIs individually

# loop over the face detections
for (i, rect) in enumerate(rects):
	# determine the facial landmarks for the face region, then convert the
	# landmark (x, y)-coordinates to a NumPy array
	shape = predictor(gray, rect)
	shape = face_utils.shape_to_np(shape)

	#loop over the face parts individually
	for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
		# clone the original image so we can draw on it, then display the
		# name of the face part on the image
		clone = image.copy()
		cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
			0.7, (0, 0, 255), 2)

		# loop over the subset of facial landmarks, drawing the specific
		# face part
		for (x, y) in shape[i:j]:
			cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)
	# For each face region, we determine the facial landmarks of the ROI and
	# convert the 68 points into a NumPy array (Lines 38 and 39).
	# Then, for each of the face parts, we loop over them and on Line 42.
	# We draw the name/label of the face region on Lines 46 and 47, then draw
	# each of the individual facial landmarks as circles on Lines 51 and 52


	# extract the ROI of the face region as a seperate image
	(x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
	roi = image[y:y+h, x:x+w]
	roi = imutils.resize(roi, width = 250, inter = cv2.INTER_CUBIC)

	# show the particular face part
	cv2.imshow("ROI", roi)
	cv2.imshow("Image", clone)
	cv2.waitKey(0)

# visualize all facial landmarks with a trasparent overlay
output = face_utils.visualize_facial_landmarks(image, shape)
cv2.imshow("Image", output)
cv2.waitKey(0)

# computing the bounding box of the region is handled on line 61.
# using NumPy array slicing we can extract the ROI on line 62.
# This ROI is then resized to have a width of 250 pixels so we can better
# visualize it (line 63).
# Lines 66-68 display the individual face region to our screen.
# Lines 71-73 then apply the `visualize_facial_landmarks` function to create
# a transparent overlay for each facial part.