# USAGE
# python mask_rcnn_video.py --input videos/cats_and_dogs.mp4 --output output/cats_and_dogs_output.avi --mask-rcnn mask-rcnn-coco

# import the necessary packages
from __future__ import print_function
import sys
import matplotlib.pyplot as plt
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import copy



###------MY idea 

# Global vars:
WIDTH = 1000   # 700
STEP = 20    # 16    # this is important param to filter out moving pixel, bigger the better
QUIVER = (0, 0, 255) # (255, 100, 0)  # show in RED


def draw_flow(img, flow, step=STEP):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    print ('fx_shape', np.shape(fx))
    print ('fy_shape', np.shape(fy))
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, QUIVER)
    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis, fx, fy

#### ENDS

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input video file")
ap.add_argument("-o", "--output", required=True,
	help="path to output video file")
ap.add_argument("-m", "--mask-rcnn", required=True,
	help="base path to mask-rcnn directory")
#ap.add_argument("-v", "--visualize", type=int, default=0,
#	help="whether or not we are going to visualize each instance")
ap.add_argument("-c", "--confidence", type=float, default=0.4,  # 0.5
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="minimum threshold for pixel-wise mask segmentation")
args = vars(ap.parse_args())

# load the COCO class labels our Mask R-CNN was trained on
labelsPath = os.path.sep.join([args["mask_rcnn"],
	"object_detection_classes_coco.txt"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

# derive the paths to the Mask R-CNN weights and model configuration
#weightsPath = os.path.sep.join([args["mask_rcnn"],
#	"frozen_inference_graph.pb"])
weightsPath = os.path.sep.join([args["mask_rcnn"],
	"mask_rcnn_person_0006.h5"])
configPath = os.path.sep.join([args["mask_rcnn"],
	"mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"])

# load our Mask R-CNN trained on the COCO dataset (90 classes)
# from disk
print("[INFO] loading Mask R-CNN from disk...")
net = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)

# initialize the video stream and pointer to output video file
vs = cv2.VideoCapture(args["input"])
writer = None

# try to determine the total number of frames in the video file
try:
	fn = sys.argv[1]

	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
	print("[INFO] {} total frames in video".format(total))

# an error occurred while trying to determine the total
# number of frames in the video file
except IndexError:

	fn = 0
	print("[INFO] could not determine # of frames in video")
	total = -1

# loop over frames from the video file stream
while True:
    # read the next frame from the file
	(grabbed, frame) = vs.read()

##-----MY Changes
	prev_frame = imutils.resize(frame, width=WIDTH)
	prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
	print ('prev_frame_shape', np.shape(prev_frame))
#### ENDS------

	# if the frame was not grabbed, then we have reached the end
	# of the stream
	if not grabbed:
		break
	# construct a blob from the input frame and then perform a
	# forward pass of the Mask R-CNN, giving us (1) the bounding box
	# coordinates of the objects in the image along with (2) the
	# pixel-wise segmentation for each specific object
	
	#gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	#print ('gray_frame', np.shape(gray_frame))

	blob = cv2.dnn.blobFromImage(frame, swapRB=True, crop=False)
	#print ('blob_shape', np.shape(blob))
	net.setInput(blob)
	start = time.time()
	(boxes, masks) = net.forward(["detection_out_final", "detection_masks"])

	end = time.time()
	# loop over the number of detected objects
	for i in range(0, boxes.shape[2]):
		# extract the class ID of the detection along with the
		# confidence (i.e., probability) associated with the
		# prediction
		classID = int(boxes[0, 0, i, 1])
		confidence = boxes[0, 0, i, 2]

		# filter out weak predictions by ensuring the detected
		# probability is greater than the minimum probability
		if confidence > args["confidence"]:
			# scale the bounding box coordinates back relative to the
			# size of the frame and then compute the width and the
			# height of the bounding box
			(H, W) = frame.shape[:2]
			box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
			(startX, startY, endX, endY) = box.astype("int")
			boxW = endX - startX
			boxH = endY - startY

			# extract the pixel-wise segmentation for the object,
			# resize the mask such that it's the same dimensions of
			# the bounding box, and then finally threshold to create
			# a *binary* mask
			mask = masks[i, classID]
			mask = cv2.resize(mask, (boxW, boxH),
				interpolation=cv2.INTER_NEAREST)
			mask = (mask > args["threshold"])

			# extract the ROI of the image but *only* extracted the
			# masked region of the ROI
			roi = frame[startY:endY, startX:endX][mask]

###------------My changes

		# check to see if are going to visualize how to extract the
			# masked region itself
			#if args["visualize"] > 0:
			#	# convert the mask from a boolean to an integer mask with
			#	# to values: 0 or 255, then apply the mask
			#	visMask = (mask * 255).astype("uint8")
			#	instance = cv2.bitwise_and(roi, roi, mask=visMask)
#
#				# show the extracted ROI, the mask, along with the
#				# segmented instance
#				cv2.imshow("ROI", roi)
#				cv2.imshow("Mask", visMask)
#				cv2.imshow("Segmented", instance)
#


#			# now, extract *only* the masked region of the ROI by passing
#			# in the boolean mask array as our slice condition
#			roi = roi[mask]

#########  ---- ENds.-------------------------------


			# grab the color used to visualize this particular class,
			# then create a transparent overlay by blending the color
			# with the ROI
			color = COLORS[classID]
			blended = ((0.4 * color) + (0.6 * roi)).astype("uint8")

			# store the blended ROI in the original frame
			frame[startY:endY, startX:endX][mask] = blended
			#print ('frame_size', np.shape(frame))
			#print ('blended_size', np.shape(blended))

			# draw the bounding box of the instance on the frame
			color = [int(c) for c in color]
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				color, 2)

			# draw the predicted label and associated probability of
			# the instance segmentation on the frame
			text = "{}: {:.4f}".format(LABELS[classID], confidence)
			cv2.putText(frame, text, (startX, startY - 5),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


			####----New idea ---
			## Once I get the mask of ROI, lets get the optical flow of the mask in each frame and from there we can catagorise activities
			# once we get the optical flow we can create another bounding box on the original frame with TAG : Picking / Not Picking
			
			clone = copy.copy(frame)
			curr_frame = copy.copy(clone)
			curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
			curr_frame = imutils.resize(curr_frame, width=WIDTH)
			#vel_x = None
			#vel_y = None
			#visual = None
			print ('clone_frame', np.shape(clone))
			print ('curr_frame', np.shape(curr_frame))
			flow = cv2.calcOpticalFlowFarneback(prev_frame, curr_frame, flow=None, pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.5, flags=0)
			visual, vel_x, vel_y = draw_flow(curr_frame, flow)
			#vel_x_list = []
			#vel_y_list = []
			#vel_y_list = vel_y_list.append(vel_y)
			#vel_x_list = vel_x_list.append(vel_x)
			#print ('vel_x', vel_x)
			cv2.imshow('visual', visual)
			prev_frame = curr_frame

			### Idea ENDS--------------

			# show the output image
			cv2.imshow("Output", clone)
			cv2.waitKey(10000)

			

	# check if the video writer is None
	if writer is None:
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30,
			(frame.shape[1], frame.shape[0]), True)

		# some information on processing single frame
		if total > 0:
			elap = (end - start)
			print("[INFO] single frame took {:.4f} seconds".format(elap))
			print("[INFO] estimated total time to finish: {:.4f}".format(
				elap * total))

	# write the output frame to disk
	writer.write(frame)

# release the file pointers
print("[INFO] cleaning up...")
writer.release()
vs.release()
