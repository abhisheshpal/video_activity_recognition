#!/usr/bin/env python3
# USAGE

# NEW USAGE
# python mask_rcnn-Copy1.py --output output/picking_image_out.avi --mask-rcnn mask-rcnn-coco --visualize 1

# import the necessary packages
import numpy as np
import argparse
import random
import time
import cv2
import imutils
import os
import copy
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.stats import mode
from skimage import draw
import math
# from optical_flow_class import OpticalFlowCalculator 
# from walk_assistance_optical_flow import OptFlow
#import matplotlib.mlab as mlab
# from scipy.spatial import distance
# from scipy.stats import norm
# from numpy.linalg import inv
# from sklearn.mixture import GaussianMixture

###------MY idea 
# Global vars:
WIDTH = 1000   # 700
STEP = 8    # 16    # this is important param to filter out moving pixel, bigger the better
QUIVER = (0, 0, 255) # (255, 100, 0)  # show in RED


###------------Apply for getting the velocity in m/sec-----------

# def processBytes(self, rgb_bytes, distance=None, timestep=1):
# 	'''
# 	Processes one frame of RGB bytes, returning summed X,Y flow.
# 	Optional inputs are:
# 	  distance - distance in meters to image (focal length) for returning flow in meters per second
# 	  timestep - time step in seconds for returning flow in meters per second
# 	 '''

# 	frame = np.frombuffer(rgb_bytes, np.uint8)
# 	frame = np.reshape(frame, (self.size[1], self.size[0], 3))
# 	return self.processFrame(frame, distance, timestep)


def my_vel_function_draw_flow(img, flow, step=STEP, motion_threshold=1., distance_meters= 0.00193, timestep_seconds=1., dimsize_pixels= 0.0000003, perspective_angle=1.):
	xsum, ysum = 0,0
	xvel, yvel = 0,0
	prev_time = None
	count =  (flow.shape[0] * flow.shape[1]) / step**2

	h, w = img.shape[:2]
	y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1).astype(int)

	fx, fy = flow[y, x].T
	xsum += fx
	ysum += fy

	x_average_velocity_pixels_per_second = xsum / count / timestep_seconds
	y_average_velocity_pixels_per_second = ysum / count / timestep_seconds

	distance_pixels = (dimsize_pixels/2) / math.tan(perspective_angle/2)
	pixels_per_meter = distance_pixels / distance_meters

	xvel = x_average_velocity_pixels_per_second / pixels_per_meter
	yvel = y_average_velocity_pixels_per_second / pixels_per_meter

	lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
	lines = np.int32(lines + 0.5)
	vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
	cv2.polylines(vis, lines, 0, QUIVER)

	for (x1, y1), (_x2, _y2) in lines:
		cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)

	# Default to system time if no timestep
	curr_time = time.time()
	if not timestep_seconds:
		timestep_seconds = (curr_time - prev_time) if prev_time else 1
		prev_time = curr_time
        
# 	if perspective_angle == 0 and distance_meters == 0:
# 		return x_average_velocity_pixels_per_second
# 	else:
# 		return xvel, yvel

# 	return xvel \ if perspective_angle and distance_meters \ else x_average_velocity_pixels_per_second

	return xsum, ysum, timestep_seconds, xvel,  yvel, x_average_velocity_pixels_per_second, y_average_velocity_pixels_per_second

###------------END ----Apply for getting the velocity in m/sec-----------



def draw_flow(img, flow, step=STEP, motion_threshold=1.):
	h, w = img.shape[:2]
	y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1).astype(int)

	fx, fy = flow[y, x].T
    
	lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
	lines = np.int32(lines + 0.5)
	vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
	cv2.polylines(vis, lines, 0, QUIVER)
	for (x1, y1), (_x2, _y2) in lines:
		cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)

	h, w = flow.shape[:2]
	flow_neg = -flow
	#fx1, fy1 = flow_neg[:2]
	flow_neg[:,:,0] += np.arange(w)
	flow_neg[:,:,1] += np.arange(h)[:,np.newaxis]
	warp_img = cv2.remap(img, flow_neg, None, cv2.INTER_LINEAR)
	return vis, warp_img, fx, fy



# def dissimilarity_dist(res_vel, angle):
# 	for i, j in zip(res_vel, angle):
# 		res_dev = 0.
# 		ori_dev = 0.
# 		both_dev = 0.
# 		add_res_dev = 0.
# 		add_ori_dev = 0.
# 		both_add_dev = 0.
# 		both_di_dev = 0.
        
# 		res_dev = i - np.mean(res_vel)
# 		ori_dev = j - np.mean(angle)
# 		both_dev += res_dev * ori_dev

# 		add_res_dev += (res_dev)*(res_dev)
# 		add_ori_dev += (ori_dev)*(ori_dev)

# 		both_add_dev = 	(add_res_dev)*(add_ori_dev)
# 		both_add_dev = np.sqrt(both_add_dev)
# 		both_di_dev = 1 - (both_dev / both_add_dev)

# 	return both_di_dev, both_dev, both_add_dev 



## calculate theta 
def cal_theta(vx, vy):
    
	fig, axs = plt.subplots(2)
	fig.suptitle('histogram showing horizontal and vertical flow of each pixel')
	axs[0].hist(vx, bins = 10)
	axs[1].hist(vy, bins = 10)
	plt.show()
    
# 	mean_vx = np.mean(vx) # can be used later 
# 	mean_vy = np.mean(vy)

    # calculate the magnitude of horizontal and vertical flow
	res_velocity = np.sqrt(vx**2 + vy**2)  ## If theta is negative it means the motion is inhibitory else excitatory
	print('res_velocity', type(res_velocity))
    
    # Method 1 to calculate theta 
    # here we will cal arctan()
	if vx.all() > 0 and vy.all() >= 0:
		theta = np.arctan(vy, vx)* (180 / np.pi) % 180
	elif vx.all() > 0 and vy.all() < 0:
		theta = (np.arctan(vy, vx) + 2*np.pi) * (180 / np.pi) % 180
	elif vx.all() < 0: 
		theta = (np.arctan(vy, vx) + np.pi)* (180 / np.pi) % 180
	elif vx.all() == 0 and vy.all() > 0:
		theta = (np.pi/2)* (180 / np.pi) % 180
	elif vx.all() == 0 and vy.all() < 0:
		theta = (3*np.pi/2)* (180 / np.pi) % 180
	elif vx.all() == 0 and vy.all() == 0:
		theta = 0.

	return theta, res_velocity



## Calculate Manhalanbolis distance for getting the scatteredness 
def evaluation_mat(cluster):

	# create a gaussian model
 	# count_fetaures = np.count(cluster)
	mean_cluster = np.mean(cluster)
	min_val = np.min(cluster)
	max_val = np.max(cluster)
	R = np.abs(max_val - min_val)

	add_diff = 0
	# cal Mean absolute deviation in a cluster
	for i in range(len(cluster)):
		diff = np.abs(i - mean_cluster)
		add_diff = add_diff + diff
	mean_abs_diff = add_diff / len(cluster)


	for i in range(len(cluster)):
		diff_sqr = (i - mean_cluster) * (i - mean_cluster)
		#add_diff = add_diff + diff
	if len(cluster) == 1:
		print('cluster feature is just 1')
	else:  
		diff_mean = diff_sqr / (len(cluster) -1)

	std_dev = np.sqrt(diff_mean) 

	print ('Range of cluster', R)
	print ('mean_abs_diff', mean_abs_diff)
	print ('Std_dev', std_dev)

	return R, mean_abs_diff, std_dev #, count_fetaures

####========================================================
## ------------ Below applying HOOF method and HOG method
###============================================================
from scipy.ndimage import uniform_filter
# from skimage.feature import hog


def get_depthFlow(imPrev, imNew):
    # Should actually go much more than 1 pixel!!!
    flow = np.zeros_like(imPrev)+999
    # flow = np.repeat(flow, 2, 2)
    
    # flow[im1==im2,:]=0
    # flow[imPrev==imPrev]=4
    flow[imPrev==imNew]=4
    for x in range(1,imPrev.shape[0]):
        for y in range(1,imPrev.shape[1]):
            if flow[x,y]==999:
                flow[x,y] = np.argmin(imPrev[x-1:x+2, y-1:y+2]-imNew[x-1:x+2, y-1:y+2])
    flow[flow==999] = -2

    flowNew = np.repeat(flow[:,:,np.newaxis], 2, 2)
    flowNew[flow==0,:] = [-1,-1]
    flowNew[flow==1,:] = [-1, 0]
    flowNew[flow==2,:] = [-1, 1]
    flowNew[flow==3,:] = [ 0,-1]
    flowNew[flow==4,:] = [ 0,0]
    flowNew[flow==5,:] = [ 0, 1]
    flowNew[flow==6,:] = [ 1,-1]
    flowNew[flow==7,:] = [ 1, 0]
    flowNew[flow==8,:] = [ 1, 1]
    return flow


# def hog2image(hogArray, imageSize=[32,32],orientations=1,pixels_per_cell=(8, 8),cells_per_block=(3, 3)):


#     sy, sx = imageSize
#     cx, cy = pixels_per_cell
#     bx, by = cells_per_block

#     n_cellsx = int(np.floor(sx // cx))  # number of cells in x
#     n_cellsy = int(np.floor(sy // cy))  # number of cells in y

#     n_blocksx = (n_cellsx - bx) + 1
#     n_blocksy = (n_cellsy - by) + 1    

#     hogArray = hogArray.reshape([n_blocksy, n_blocksx, by, bx, orientations])

#     orientation_histogram = np.zeros((n_cellsy, n_cellsx, orientations))
#     for x in range(n_blocksx):
#             for y in range(n_blocksy):
#                 block = hogArray[y, x, :]
#                 orientation_histogram[y:y + by, x:x + bx, :] = block

#     radius = min(cx, cy) // 2 - 1
#     hog_image = np.zeros((sy, sx), dtype=float)
#     for x in range(n_cellsx):
#         for y in range(n_cellsy):
#             for o in range(orientations):
#                 centre = tuple([y * cy + cy // 2, x * cx + cx // 2])
#                 dx = int(radius * np.cos(float(o) / orientations * np.pi))
#                 dy = int(radius * np.sin(float(o) / orientations * np.pi))
#                 # rr, cc = draw.bresenham(centre[0] - dy, centre[1] - dx,
#                 #                         centre[0] + dy, centre[1] + dx)
#                 rr, cc = draw.line(centre[0] - dx, centre[1] - dy,\
#                                         centre[0] + dx, centre[1] + dy)  
#                 hog_image[rr, cc] += orientation_histogram[y, x, o]

#     return hog_image


# def hof(flow, orientations=1, pixels_per_cell=(8, 8),
#         cells_per_block=(3, 3), visualise=True, normalise=True, motion_threshold=1.):

#     """Extract Histogram of Optical Flow (HOF) for a given image.

#     Key difference between this and HOG is that flow is MxNx2 instead of MxN


#     Compute a Histogram of Optical Flow (HOF) by

#         1. (optional) global image normalisation
#         2. computing the dense optical flow
#         3. computing flow histograms
#         4. normalising across blocks
#         5. flattening into a feature vector

#     Parameters
#     ----------
#     Flow : (M, N) ndarray
#         Input image (x and y flow images).
#     orientations : int
#         Number of orientation bins.
#     pixels_per_cell : 2 tuple (int, int)
#         Size (in pixels) of a cell.
#     cells_per_block  : 2 tuple (int,int)
#         Number of cells in each block.
#     visualise : bool, optional
#         Also return an image of the hof.
#     normalise : bool, optional
#         Apply power law compression to normalise the image before
#         processing.
#     static_threshold : threshold for no motion

#     Returns
#     -------
#     newarr : ndarray
#         hof for the image as a 1D (flattened) array.
#     hof_image : ndarray (if visualise=True)
#         A visualisation of the hof image.

#     References
#     ----------
#     * http://en.wikipedia.org/wiki/Histogram_of_oriented_gradients

#     * Dalal, N and Triggs, B, Histograms of Oriented Gradients for
#       Human Detection, IEEE Computer Society Conference on Computer
#       Vision and Pattern Recognition 2005 San Diego, CA, USA

#     """
#     flow = np.atleast_2d(flow)

#     """ 
#     -1-
#     The first stage applies an optional global image normalisation
#     equalisation that is designed to reduce the influence of illumination
#     effects. In practice we use gamma (power law) compression, either
#     computing the square root or the log of each colour channel.
#     Image texture strength is typically proportional to the local surface
#     illumination so this compression helps to reduce the effects of local
#     shadowing and illumination variations.
#     """

#     if flow.ndim < 3:
#         raise ValueError("Requires dense flow in both directions")

#     if normalise:
#         flow = np.sqrt(flow)

#     """ 
#     -2-
#     The second stage computes first order image gradients. These capture
#     contour, silhouette and some texture information, while providing
#     further resistance to illumination variations. The locally dominant
#     colour channel is used, which provides colour invariance to a large
#     extent. Variant methods may also include second order image derivatives,
#     which act as primitive bar detectors - a useful feature for capturing,
#     e.g. bar like structures in bicycles and limbs in humans.
#     """

#     if flow.dtype.kind == 'u':
#         # convert uint image to float
#         # to avoid problems with subtracting unsigned numbers in np.diff()
#         flow = flow.astype('float')

#     gx = np.zeros(flow.shape[:2])
#     gy = np.zeros(flow.shape[:2])
#     # gx[:, :-1] = np.diff(flow[:,:,1], n=1, axis=1)
#     # gy[:-1, :] = np.diff(flow[:,:,0], n=1, axis=0)

#     gx = flow[:,:,1]
#     gy = flow[:,:,0]



#     """ 
#     -3-
#     The third stage aims to produce an encoding that is sensitive to
#     local image content while remaining resistant to small changes in
#     pose or appearance. The adopted method pools gradient orientation
#     information locally in the same way as the SIFT [Lowe 2004]
#     feature. The image window is divided into small spatial regions,
#     called "cells". For each cell we accumulate a local 1-D histogram
#     of gradient or edge orientations over all the pixels in the
#     cell. This combined cell-level 1-D histogram forms the basic
#     "orientation histogram" representation. Each orientation histogram
#     divides the gradient angle range into a fixed number of
#     predetermined bins. The gradient magnitudes of the pixels in the
#     cell are used to vote into the orientation histogram.
#     """

#     magnitude = np.sqrt(gx**2 + gy**2)
#     orientation = np.arctan2(gy, gx) * (180 / np.pi) % 180

#     sy, sx = flow.shape[:2]
#     cx, cy = pixels_per_cell
#     bx, by = cells_per_block

#     n_cellsx = int(np.floor(sx // cx))  # number of cells in x
#     n_cellsy = int(np.floor(sy // cy))  # number of cells in y

#     # compute orientations integral images
#     orientation_histogram = np.zeros((n_cellsy, n_cellsx, orientations))
#     subsample = np.index_exp[cy // 2:cy * n_cellsy:cy, cx // 2:cx * n_cellsx:cx]
#     for i in range(orientations-1):
#         #create new integral image for this orientation
#         # isolate orientations in this range

#         temp_ori = np.where(orientation < 180 / orientations * (i + 1),
#                             orientation, -1)
#         temp_ori = np.where(orientation >= 180 / orientations * i,
#                             temp_ori, -1)
#         # select magnitudes for those orientations
#         cond2 = (temp_ori > -1) * (magnitude > motion_threshold)
#         temp_mag = np.where(cond2, magnitude, 0)

#         temp_filt = uniform_filter(temp_mag, size=(cy, cx))
#         orientation_histogram[:, :, i] = temp_filt[subsample]
#     print ('temp_mag', temp_mag)
#     print ('temp_ori', temp_ori)
#     ''' Calculate the no-motion bin '''
#     temp_mag = np.where(magnitude <= motion_threshold, magnitude, 0)

#     temp_filt = uniform_filter(temp_mag, size=(cy, cx))
#     orientation_histogram[:, :, -1] = temp_filt[subsample]

#     # now for each cell, compute the histogram
#     hof_image = None

#     if visualise:
#         from skimage import draw

#         radius = min(cx, cy) // 2 - 1
#         hof_image = np.zeros((sy, sx), dtype=float)
#         for x in range(n_cellsx):
#             for y in range(n_cellsy):
#                 for o in range(orientations-1):
#                     centre = tuple([y * cy + cy // 2, x * cx + cx // 2])
#                     dx = int(radius * np.cos(float(o) / orientations * np.pi))
#                     dy = int(radius * np.sin(float(o) / orientations * np.pi))
#                     rr, cc = draw.line(centre[0] - dy, centre[1] - dx,
#                                             centre[0] + dy, centre[1] + dx)
#                     hof_image[rr, cc] += orientation_histogram[y, x, o]

#     """
#     The fourth stage computes normalisation, which takes local groups of
#     cells and contrast normalises their overall responses before passing
#     to next stage. Normalisation introduces better invariance to illumination,
#     shadowing, and edge contrast. It is performed by accumulating a measure
#     of local histogram "energy" over local groups of cells that we call
#     "blocks". The result is used to normalise each cell in the block.
#     Typically each individual cell is shared between several blocks, but
#     its normalisations are block dependent and thus different. The cell
#     thus appears several times in the final output vector with different
#     normalisations. This may seem redundant but it improves the performance.
#     We refer to the normalised block descriptors as Histogram of Oriented
#     Gradient (hog) descriptors.
#     """

#     n_blocksx = (n_cellsx - bx) + 1
#     n_blocksy = (n_cellsy - by) + 1
#     normalised_blocks = np.zeros((n_blocksy, n_blocksx,
#                                   by, bx, orientations))

#     for x in range(n_blocksx):
#         for y in range(n_blocksy):
#             block = orientation_histogram[y:y+by, x:x+bx, :]
#             eps = 1e-5
#             normalised_blocks[y, x, :] = block / np.sqrt(block.sum()**2 + eps)

#     """
#     The final step collects the hof descriptors from all blocks of a dense
#     overlapping grid of blocks covering the detection window into a combined
#     feature vector for use in the window classifier.
#     """

#     if visualise:
#         return normalised_blocks.ravel(), hof_image
#     else:
#         return normalised_blocks.ravel()

####====================================================================


# Create mahanbolis distance 





# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--image", required=True,
#	help="path to input image")

ap.add_argument("-o", "--output", required=True,
	help="path to output video file")
#ap.add_argument("-o2", "--output2", required=True,
#	help="path to output video2 file")
ap.add_argument("-m", "--mask-rcnn", required=True,
	help="base path to mask-rcnn directory")
ap.add_argument("-v", "--visualize", type=int, default=0,
	help="whether or not we are going to visualize each instance")
ap.add_argument("-c", "--confidence", type=float, default=0.4,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="minimum threshold for pixel-wise mask segmentation")
args = vars(ap.parse_args())

# load the COCO class labels our Mask R-CNN was trained on
labelsPath = os.path.sep.join([args["mask_rcnn"],
	"object_detection_classes_coco.txt"])
LABELS = open(labelsPath).read().strip().split("\n")

# load the set of colors that will be used when visualizing a given
# instance segmentation
colorsPath = os.path.sep.join([args["mask_rcnn"], "colors.txt"])
COLORS = open(colorsPath).read().strip().split("\n")
COLORS = [np.array(c.split(",")).astype("int") for c in COLORS]
COLORS = np.array(COLORS, dtype="uint8")

# derive the paths to the Mask R-CNN weights and model configuration
weightsPath = os.path.sep.join([args["mask_rcnn"],
	"frozen_inference_graph.pb"])
configPath = os.path.sep.join([args["mask_rcnn"],
	"mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"])

# load our Mask R-CNN trained on the COCO dataset (90 classes)
# from disk
print("[INFO] loading Mask R-CNN from disk...")
net = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)

# load our input image and grab its spatial dimensions
mags = []
angs = np
res_velo = []
flow_oris = []

# cam = cv2.VideoCapture('/home/abhishesh01/video_segmentation/workforJUNE2020/mask-rcnn_for_postprocessing/videos/unloading_videos/norway_orchard_unloading_2.mp4')
cam = cv2.VideoCapture('/home/abhishesh01/video_segmentation/workforJUNE2020/mask-rcnn_for_postprocessing/videos/picking2.mp4')
ret, frame = cam.read()
#prev_frame = imutils.resize(frame, width=WIDTH)
prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
prev_frame = np.float32(prev_frame)/255.0
#cv2.imshow('prev_frame', prev_frame)
print ('prev_frame_shape', np.shape(prev_frame))


### MY Idea ---------- >>> Apply Discrete cosine transform (DCT)for video / image compression for less storage  &&  
# We Apply DCT only on gray scale video it means we have to apply this on OPTICAL FLOW only
##NOTE : Also apply FFT transformation and then compare with DCT and show why should we use DCT


###---END Idea
count = 0
writer = None
#writer2 = None

# try to determine the total number of frames in the video file
try:
#	fn = sys.argv[1]

	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(cam.get(prop))
	print("[INFO] {} total frames in video".format(total))

# an error occurred while trying to determine the total
# number of frames in the video file
except IndexError:

	fn = 0
	print("[INFO] could not determine # of frames in video")
	total = -1
 

if (cam.isOpened()== False):
  print("Error opening video stream or file")

while (cam.isOpened()):
	count +=1
	print ('frame_count', count)

	if ret == True:
	#if ret == True:
		ret, image = cam.read()
		#image = cv2.imread(args["image"])
		(H, W) = image.shape[:2]

		# construct a blob from the input image and then perform a forward
		# pass of the Mask R-CNN, giving us (1) the bounding box coordinates
		# of the objects in the image along with (2) the pixel-wise segmentation
		# for each specific object
		blob = cv2.dnn.blobFromImage(image, swapRB=True, crop=False)
		net.setInput(blob)
		start = time.time()
		(boxes, masks) = net.forward(["detection_out_final", "detection_masks"])
		end = time.time()

		# show timing information and volume information on Mask R-CNN
		print("[INFO] Mask R-CNN took {:.6f} seconds".format(end - start))
		print("[INFO] boxes shape: {}".format(boxes.shape))
		print("[INFO] masks shape: {}".format(masks.shape))

		# loop over the number of detected objects
		for i in range(0, boxes.shape[2]):
			# extract the class ID of the detection along with the confidence
			# (i.e., probability) associated with the prediction
			classID = int(boxes[0, 0, i, 1])
			confidence = boxes[0, 0, i, 2]

			# to just get classID of person
			if classID != 0:
				classID = int(boxes[0, 0, 0, 1])
				confidence = boxes[0, 0, i, 2]
			# filter out weak predictions by ensuring the detected probability
			# is greater than the minimum probability
			if confidence > args["confidence"]:
				# clone our original image so we can draw on it
				clone = image.copy()
				# scale the bounding box coordinates back relative to the
				# size of the image and then compute the width and the height
				# of the bounding box
				box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
				(startX, startY, endX, endY) = box.astype("int")
				boxW = endX - startX
				boxH = endY - startY

				# extract the pixel-wise segmentation for the object, resize
				# the mask such that it's the same dimensions of the bounding
				# box, and then finally threshold to create a *binary* mask
				mask = masks[i, classID]
				#print ('mask', mask.shape)
				mask = cv2.resize(mask, (boxW, boxH),
					interpolation=cv2.INTER_NEAREST)
				mask = (mask > args["threshold"])
				
				# extract the ROI of the image
				roi = clone[startY:endY, startX:endX]
				

				############---------------- Idea to know only PERSON class------------
				#visMask = (mask * 255).astype("uint8")
				#roi_mask = cv2.bitwise_and(roi, roi, mask=visMask)
				#cv2.imshow('roi_mask', roi_mask)
				#cv2.waitKey(10000)
				#roi_mask = roi_mask.astype(int)
				
				#################----------END--------------------------------

				# check to see if are going to visualize how to extract the
				# masked region itself
				if args["visualize"] > 0:
					# convert the mask from a boolean to an integer mask with
					# to values: 0 or 255, then apply the mask
					visMask = (mask * 255).astype("uint8")
					instance = cv2.bitwise_and(roi, roi, mask=visMask)
					#print('segmented', instance.shape)
					#print ('roi', roi.shape)
					#print('mask', mask.shape)
					# show the extracted ROI, the mask, along with the
					# segmented instance
					#cv2.imshow("ROI", roi)
					#cv2.imshow("Mask", visMask)
					#cv2.imshow("Segmented", instance)

###===============================================================================================================================

				###--- Abhishesh START - create a black frame for a ref. as prev_frame to apply optical flow in 'segmented image'

###===============================================================================================================================
			
				prev_visMask = (mask * 0).astype("uint8")
				prev_instance = cv2.bitwise_and(roi, roi, mask= prev_visMask)				
				instance = cv2.cvtColor(instance, cv2.COLOR_BGR2GRAY)
				prev_instance = cv2.bitwise_and(roi, roi, mask= prev_visMask)
				prev_instance = cv2.cvtColor(prev_instance, cv2.COLOR_BGR2GRAY)
				# get the flow of roi
# 				flow_roi = cv2.calcOpticalFlowFarneback(prev_instance, instance, flow=None, pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.5, flags=0)
				flow_roi = cv2.calcOpticalFlowFarneback(prev_instance, instance, flow=None, pyr_scale=0.5, levels=3, winsize=25, iterations=3, poly_n=5, poly_sigma=1.5, flags=0)
				flow_avg = np.median (flow_roi , axis =(0,1)) # [x, y]   
				# The direction of motion 
				move_x =  -1 * flow_avg[0] 
				move_y =  -1 * flow_avg[1] 
                
#               to normalize flow to coap up with intensity 
# 				flow_roi = np.sqrt(flow_roi)

				# create a depth image of the foreground detected object				
				depth_flow = get_depthFlow(prev_instance, instance)
				# get the overall mag and angle of the whole image at each pixel level
				mag, ang = cv2.cartToPolar(flow_roi[..., 0], flow_roi[..., 1]) #, angleInDegrees=True)

                # Store magnitude for all the objects identified      
# 				mags = np.vstack([mag])
# 				angs = np.vstack([ang])

# 				ang_dir_median = np.median(ang , axis =(0,1)) # [x, y]   
# 				# The direction of motion
# 				ang_x =  -1 * flow_avg[0] 
# 				ang_y =  -1 * flow_avg[1]

                
				#if ang.any() < 0:
				#	ang = ang + 360
# 				ang = ang * (2 * np.pi % 180)
				
				mag_nonzero = np.count_nonzero(mag)
				ang_nonzero = np.count_nonzero(ang)

				if (np.sum(mag)> 600):
					print('motion detected')

				if np.max(np.median(mag)) > 20 and np.max(np.median(ang)) > 39:
					print ("***the event might be picking***")
				else:
					print ("***the event might be unloading***") 
				# store mag and angle for next frames 
# 				if mag>0:
#                 		store_mag = []
# 				store_mag = store_mag.append(mag)
# 				print ("store_mag", store_mag)               
                
				#  Divide the whole roi regions into definite grids to reduce the computation 
				vis_flow_roi,vis_flow_warp, vx_flow_roi, vy_flow_roi = draw_flow(instance, flow_roi)
				# calculate velocity in meters/ sec
				xsum1, ysum1,timestep1, x_velocity_meters_per_second1, y_velocity_meters_per_second1, x_average_velocity_pixels_per_second1, y_average_velocity_pixels_per_second1 = my_vel_function_draw_flow(instance, flow_roi)				
				# cal theta(for the direction of flow) and resulatant vector of the flow
				flow_orientation, resultant_velocity = cal_theta(vx_flow_roi, vy_flow_roi)
				time_passed = end - start
				acc_hor = vx_flow_roi / time_passed
				acc_ver = vy_flow_roi / time_passed
				
				# Store orientation and res_vel of several frames here
#				flow_oris = np.vstack([flow_orientation])
#				res_velo  = np.vstack([resultant_velocity])
				# Store orientation and res_vel of several frames here
				#f = np.vstack([flow_orientation])
				#r = np.vstack([resultant_velocity])
				#f = None
				# Store orientation and res_vel of several frames here
				#if f is not None:
				f = np.vstack([flow_orientation])
				print ('shape of f', np.shape(f))
				#r = np.vstack([resultant_velocity])
				#else:
				#f = flow_orientation

#       		          # create HOG
# 				HOG = hog2image(vis_flow_roi, imageSize=[32,32],orientations=9,pixels_per_cell=(8, 8),cells_per_block=(3, 3)) 
# 				cv2.imshow('HOG_image', HOG)
				# create HOOF 				
# 				normalised, HOF_image = hof(flow_roi, orientations=12, pixels_per_cell=(8, 8), cells_per_block=(3, 3), visualise=True, normalise=False, motion_threshold=1.)
# 				cv2.imshow('HOF_image', HOF_image)                

				# get the no. of features in each bins ro get the median 
				median_ang = np.median(flow_orientation)
                
  		###=================================================================================================
				
                # The idea is to devide the optical flow into n-grids (using draw_flow) and calculate :-
                
                # METHOD 1:  Segmentation using HOOF of theta and Weight velocity method : In this method 
                    
                                
               ###==================================================================================================
                
                
				## cal HOOF of theta	
				bins_number = 12  # the [0, 360) interval will be subdivided into this

###----------------------------------------------------------
# 				# get the mag falls withing angular range 				

				motion_threshold = 1.
# 				for i in range(bins_number-1):
# 					#create new integral image for this orientation
# 					# isolate orientations in this range

# 					temp_ori = np.where(flow_orientation < 180 / bins_number * (i + 1),
# 					flow_orientation, -1)
# 					temp_ori = np.where(flow_orientation >= 180 / bins_number * i,
# 					temp_ori, -1)
# 					# select magnitudes for those orientations
# 					cond2 = (temp_ori > -1) * (resultant_velocity > motion_threshold)
# 					temp_mag = np.where(cond2, resultant_velocity, 0)
# 				''' Calculate the no-motion bin '''
# 				temp_mag = np.where(resultant_velocity <= motion_threshold, resultant_velocity, 0)
# # 				if temp_mag  
# 				plt.hist(temp_mag)
# 				plt.show()
# 				plt.hist(temp_ori)
# 				plt.show()


# 				# number of equal bins
# 				bins = np.linspace(0.0, 2 * np.pi, bins_number+1)
# 				n, _, _ = plt.hist(temp_ori, bins)
# 				bin_counts = np.unique(bins)
# 				plt.clf()
# 				width = 2 * np.pi / bins_number
# 				colors = plt.cm.viridis(n/10.)
# 				ax = plt.subplot(1, 1, 1, projection='polar')
# 				bars = ax.bar(bins[:bins_number], n, width=width, bottom=0.0, color= colors)
# # 				ax.set_ylim(0, 10)
# 				ax.set_rlabel_position(270)
# # 				print ('bins[:bins_number]', t )#bins[:bins_number])
# 				for bar in bars:
# 					bar.set_alpha(0.5)
# 				plt.title("direction plot")
# 				plt.show() 
#----------------------------------------------------------------------

				# number of equal bins
				bins = np.linspace(0.0, 2 * np.pi, bins_number+1)
				n, _, _ = plt.hist(flow_orientation, bins)
				bin_counts = np.unique(bins)
				plt.clf()
				width = 2 * np.pi / bins_number
				colors = plt.cm.viridis(n/10.)
				ax = plt.subplot(1, 1, 1, projection='polar')
				bars = ax.bar(bins[:bins_number], n, width=width, bottom=0.0, color= colors)
# 				ax.set_ylim(0, 10)
				ax.set_rlabel_position(270)
# 				print ('bins[:bins_number]', t )#bins[:bins_number])
				for bar in bars:
					bar.set_alpha(0.5)
				plt.title("direction plot")
				plt.show() 
				
				# get mean of angular orientation 
				mean_ang = np.mean(flow_orientation)
				# get the horizontal resultant velocity					
				hor_vx = np.abs(resultant_velocity*np.cos(flow_orientation))
				hor_vx_mean = np.mean(hor_vx)

				ver_vy = np.abs(resultant_velocity*np.sin(flow_orientation))
				ver_vy_mean = np.mean(ver_vy)
				
				res_mag = np.sqrt(hor_vx**2 + ver_vy**2)
				diff_mean_vel = np.abs(hor_vx_mean - ver_vy_mean)


				# Calculate Dissimilarity Distance
# 				Dist, nu , den = dissimilarity_dist(resultant_velocity, flow_orientation)
                
          			# calculate correlation distance 				
				corr_dist = (resultant_velocity * flow_orientation)**2

								
				##-----------------------------
				## cal actual orientation of the moving object
				alpha_hor = np.arccos((vx_flow_roi *resultant_velocity)/(np.abs(vx_flow_roi)) * (np.abs(resultant_velocity)))/np.size(resultant_velocity)
				alpha_hor = (alpha_hor) * 180 / np.pi % 180
				bins_number = 12  # the [0, 360) interval will be subdivided into this
				# number of equal bins
				bins = np.linspace(0.0, 2 * np.pi, bins_number+1)
				n, t, _ = plt.hist(alpha_hor, bins)
				bin_counts = np.unique(bins)
				plt.clf()
				width = 2 * np.pi / bins_number
				colors = plt.cm.viridis(n/10.)
				ax = plt.subplot(1, 1, 1, projection='polar')
				bars = ax.bar(bins[:bins_number], n, width=width, bottom=0.0, color= colors)
# 				ax.set_ylim(0, 10)
				ax.set_rlabel_position(270)
# 				print ('bins[:bins_number]', t )#bins[:bins_number])
				for bar in bars:
					bar.set_alpha(0.5)
				plt.title("direction_hor plot")
				plt.show() 
				##-----------------------------
                
                ### ---- Set criteria for classifying the activity------------
                
				if ver_vy_mean > 0 or ver_vy_mean < -0.001 or mean_ang > 60 or np.mean(res_mag) > 2 or resultant_velocity.any() >= 2:
# 				if (np.abs(ver_vy_mean) > 0.01 and (mean_ang > 60 or np.mean(res_mag) > 2)) and (np.abs(ver_vy_mean = 0 and mean_ang = 0 and np.mean(res_mag) > 2.5):# and resultant_velocity.any() >= 2:
					print ("----process is unloading----")
					print ("ver_vy_mean", ver_vy_mean)
# 					print ("resultant_velocity", resultant_velocity.any())
					print ("mean_orientation", mean_ang)
					print ("magnitide of res_velocities", np.mean(res_mag))
                # elif # and resultant_velocity.any() >= 2:
                    
				else:
					print ("----process is picking----")
					print ("ver_vy_mean", ver_vy_mean)
# 					print ("resultant_velocity", resultant_velocity.any())
					print ("mean_orientation", mean_ang)
					print ("magnitide of res_velocities", np.mean(res_mag))
					
				figure, axes = plt.subplots(nrows=2, ncols=2)
				axes[0, 0].plot(hor_vx, '-', linewidth=0.3, markersize=1)
				axes[0, 0].set_xlabel('feature points');
				axes[0, 0].set_ylabel('velocity(pixel/hz');
				axes[0, 0].set_title('hor_res_vx velocity of flow')

				axes[0, 1].plot(vx_flow_roi, '-', linewidth=0.3, markersize=1)
				axes[0, 1].set_xlabel('feature points');
				axes[0, 1].set_ylabel('velocity(pixel/hz');
				axes[0, 1].set_title('vx velocity of flow')

				axes[1, 0].plot(ver_vy, '-', linewidth=0.3, markersize=1)
				axes[1, 0].set_xlabel('feature points');
				axes[1, 0].set_ylabel('velocity(pixel/hz');
				axes[1, 0].set_title('ver_res_vy velocity of flow')


				axes[1, 1].plot(vy_flow_roi, '-', linewidth=0.3, markersize=1)
				axes[1, 1].set_xlabel('feature points');
				axes[1, 1].set_ylabel('velocity(pixel/hz');
				axes[1, 1].set_title('vx velocity of flow')

				figure, axes = plt.subplots(nrows=2) #, ncols=0)
				axes[0].plot(resultant_velocity, '-', linewidth=0.3, markersize=1)
				axes[0].set_xlabel('feature points');
				axes[0].set_ylabel('velocity(pixel/hz');
				axes[0].set_title('res_velocity velocity of flow')

				axes[1].plot(flow_orientation, '-', linewidth=0.3, markersize=1)
				axes[1].set_xlabel('feature points');
				axes[1].set_ylabel('Angle');
				axes[1].set_title('Angular orientation of motion of the flow')

                
# 				print('D has data', len(D))
				## create a gaussian mixture model 
# 				plt.plot(D, '*') #, linewidth=0.3, markersize=1)
# 				#plt.plot(D[:,0], D[:,1], 'bx')
# 				plt.axis('equal')
# 				plt.show()
# 				plt.scatter(D)
# 				plt.show()

# 				gmm = GaussianMixture(n_components=2)
# 				gmm.fit(D)

# 				print('gmm.means_', gmm.means_)
# 				#print('\n')
# 				print('gmm.covariances_', gmm.covariances_)

# 				X, Y = np.meshgrid(np.linspace(-1, 8), np.linspace(-1,8))
# 				XX = np.array([X.ravel(), Y.ravel()]).T
# 				Z = gmm.score_samples(XX)
# 				Z = Z.reshape((50,50))
# 				plt.contour(X, Y, Z)
# 				plt.scatter(D)
# 				plt.show()



  				###=================================================================================================
				
                # METHOD 2:  Segmentation using HOG  : In this method a separate m-bins histogram of angle strenght and magnitude strenght
                # is create.  
                
                # 
               ###==================================================================================================


	
				###=================================================================================================
				
                # METHOD 2: Segmentation using K-mean clustering 
                
                # Apply K-mean clustering on res_vel_roi and flow_orientation to identify valid and unvalid values
                # Use sklearn clustering method
                                
               ###==================================================================================================
				
                # for angular cluster 
				kmeans = KMeans(n_clusters=2).fit(ang)
				kmean_sklearn_distances_ang = np.column_stack([np.sum((ang - center)**2, axis=1)**0.5 for center in kmeans.cluster_centers_])
				# for mag cluster
				kmeans = KMeans(n_clusters=2).fit(mag)
				kmean_sklearn_distances_mag = np.column_stack([np.sum((mag - center)**2, axis=1)**0.5 for center in kmeans.cluster_centers_])

				
				# Using numpy based k-mean clustering method                
				# Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
				criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
				# Set flags (Just to avoid line break in the code)
				flags = cv2.KMEANS_RANDOM_CENTERS
				# Apply KMeans
				#compactnes,label,center = cv2.kmeans(res_vel_roi,3,None,criteria,10,flags)
				#compactnes,label,center = cv2.kmeans(res_vel_roi_warp,3,None,criteria,10,flags)
				compactnes,label,center = cv2.kmeans(ang,2,None,criteria,10,flags)
				compactness,labels,centers = cv2.kmeans(mag,2,None,criteria,10,flags)
				
				#print ('labels', cv2.kmeans.labels)
				# Now separate the data, Note the flatten()
				clust_mag_1 = mag[labels.ravel()==0]
				clust_mag_2 = mag[labels.ravel()==1]
# 				C = resultant_velocity[labels.ravel()==2]
				#D = resultant_velocity[labels.ravel()==3]
				#E = resultant_velocity[labels.ravel()==4]

				clust_ang_1 = ang[label.ravel()==0]
				clust_ang_2 = ang[label.ravel()==1]
# 				Ct = flow_orientation[label.ravel()==2]

				## Calculate mean of each cluster 
				mean_clust_mag_1 = np.mean(clust_mag_1)
				mean_clust_mag_2 = np.mean(clust_mag_2)
				print ('mean_clust_mag_1', mean_clust_mag_1)
				print ('mean_clust_mag_2', mean_clust_mag_2)

				# evaluation of each cluster
				err_clust_mag_1 = evaluation_mat(clust_mag_1)
				err_clust_mag_2 = evaluation_mat(clust_mag_2)

				# Plot the data



				figure, axes = plt.subplots(nrows=2, ncols=2)
				axes[0, 0].plot(clust_mag_1, '*', linewidth=0.3, markersize=1)
				axes[0, 0].set_xlabel('feature points');
				axes[0, 0].set_ylabel('vector displacement in pixel');
				axes[0, 0].set_title('Cluster 1 of mag of flow')

				axes[0, 1].plot(clust_mag_2, '*', linewidth=0.3, markersize=1)
				axes[0, 1].set_xlabel('feature points');
				axes[0, 1].set_ylabel('vector displacement in pixel');
				axes[0, 1].set_title('Cluster 2 of mag of flow')


				axes[1, 0].plot(clust_ang_1, '*', linewidth=0.3, markersize=1)
				axes[1, 0].set_xlabel('feature points');
				axes[1, 0].set_ylabel('angle in degree');
				axes[1, 0].set_title('Cluster 1 of angle of flow')
				
				axes[1, 1].plot(clust_ang_2, '*', linewidth=0.3, markersize=1)
				axes[1, 1].set_xlabel('feature points');
				axes[1, 1].set_ylabel('angle in degree');
				axes[1, 1].set_title('Cluster 2 of angle of flow')


# 				axes[2, 0].plot(C, '*', linewidth=0.3, markersize=1)
# 				axes[2, 0].set_xlabel('feature points');
# 				axes[2, 0].set_ylabel('velocity(pixel/hz)');
# 				axes[2, 0].set_title('Cluster C of resultant velocity of flow')

# 				axes[2, 1].plot(Ct, '*', linewidth=0.3, markersize=1)
# 				axes[2, 1].set_xlabel('feature points');
# 				axes[2, 1].set_ylabel('velocity(pixel/hz)');
# 				axes[2, 1].set_title('Cluster Ct of angle of flow')

				#plt.plot(vx_flow_roi, '-', linewidth=0.3, markersize=1)
				#plt.scatter(centers[:,0],centers[:,0],s = 80,c = 'y', marker = 's')
				#plt.xlabel('X'),plt.ylabel('Y')
				figure.tight_layout()
				plt.show(block=True)


				#cost_fun = np.sqrt(((vx_flow_roi)*(vx_flow_roi) + (vy_flow_roi)*(vy_flow_roi))* theta)
				#mean_res_vel_roi = np.mean(resultant_velocity)
				#print ('mean_res_vel_roi', mean_res_vel_roi)
				#print ('cost_funtype', type(cost_fun))
				#res_vel_roi = list()
				#res_vel_roi = cost_fun
				#print ('resultant_velocity_shape', resultant_velocity.shape)
				#cv2.imshow('vis_flow_roi', vis_flow_roi)
				
                
				#plt.plot('flow_x', vx_flow_roi[0:1], vy_flow_roi[0:])
				#plt.show()
				#plt.plot('flow_y', vy_flow_roi[0:1], vy_flow_roi[0:])
				#plt.show()
				#cv2.imshow("Non_Segmented", prev_instance)
				#print ("Segmented", instance.shape)
				#print ("Non_Segmented", prev_instance.shape)
				prev_instance = instance							
				
				# Store orientation and res_vel of several frames here
# 				flow_oris = np.hstack([f[i-1], f[i]])
# 				print ('shape of fLOW_ORIS', np.shape(f))
				#res_velo  = np.vstack([r])
				###---------END ----------------

				# now, extract *only* the masked region of the ROI by passing
				# in the boolean mask array as our slice condition
				roi = roi[mask]


###=============================================================================================================
                    
                ####----Abhishesh START--- New idea to apply optical flow on whole frame not just on ROI ---

###=================================================================================================================

#----------------------- check if any event is happening in whole image---------------------- 

				## Once I get the mask of ROI, lets get the optical flow of the mask in each frame and from there we can catagorise activities
				# once we get the optical flow we can create another bounding box on the original frame with TAG : Picking / Not Picking

				curr_frame = copy.copy(clone)
				#curr_frame = imutils.resize(curr_frame, width=WIDTH)
				#test_f = curr_frame.copy()
				curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
				curr_frame = np.float32(curr_frame)/255.0
				#print ('clone_frame', np.shape(clone))
				print ('curr_frame', np.shape(curr_frame))

				#flow = cv2.calcOpticalFlowFarneback(prev_frame, curr_frame, flow=None, pyr_scale=0.2, levels=5, winsize=25, iterations=5, poly_n=7, poly_sigma=1.5, flags=0)
				flow = cv2.calcOpticalFlowFarneback(prev_frame, curr_frame, flow=None, pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.5, flags=0)
				visual, visual_warp, vel_x, vel_y = draw_flow(curr_frame, flow)
				
				#res_vel_roi = np.sqrt((vel_x)*(vel_x) + (vel_y)*(vel_y))
				#print ('res_vel_roi', res_vel_roi)
				#print ('vel_x', vel_x)
				#print ('vel_y', vel_y)
				#cv2.imshow('visual', visual)
				prev_frame = curr_frame
				#print ('visual', np.shape(visual))




###=======================================================================================================================				
				###---------END ----------------
###=======================================================================================================================


				# randomly select a color that will be used to visualize this
				# particular instance segmentation then create a transparent
				# overlay by blending the randomly selected color with the ROI
				color = random.choice(COLORS)
				blended = ((0.4 * color) + (0.6 * roi)).astype("uint8")
				#print ('blended', blended.shape)			
				# store the blended ROI in the original image
				clone[startY:endY, startX:endX][mask] = blended

				# draw the bounding box of the instance on the image
				color = [int(c) for c in color]
				cv2.rectangle(clone, (startX, startY), (endX, endY), color, 2)

				# draw the predicted label and associated probability of the
				# instance segmentation on the image
				text = "{}: {:.4f}".format(LABELS[classID], confidence)
				cv2.putText(clone, text, (startX, startY - 5),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


				# show the output image
				cv2.imshow("Output", clone)
				cv2.imshow('vis_flow_roi', vis_flow_roi)
				cv2.imshow('warp_flow', vis_flow_warp)




######==================================
###create plots
###======================================

# 				figure, axes = plt.subplots(nrows=3, ncols=2)
# 				axes[0, 0].plot(resultant_velocity, '*', linewidth=0.3, markersize=1)
# 				axes[0, 0].set_xlabel('feature points');
# 				axes[0, 0].set_ylabel('velocity(pixel/hz)');
# 				axes[0, 0].set_title('resultant velocity of flow')
# 				
# 				axes[0, 1].plot(flow_orientation, '-', linewidth=0.3, markersize=1)
# 				axes[0, 1].set_xlabel('angle');
# 				axes[0, 1].set_ylabel('feature points)');
# 				axes[0, 1].set_title('resultant direction of flow')

# 				axes[1, 0].plot(vx_flow_roi, '-', linewidth=0.3, markersize=1)
# 				axes[1, 0].set_xlabel('feature points');
# 				axes[1, 0].set_ylabel('velocity(pixel/hz)');
# 				axes[1, 0].set_title('horizontal velocity of flow')

# 				axes[1, 1].plot(vy_flow_roi, '-', linewidth=0.3, markersize=1)
# 				axes[1, 1].set_xlabel('feature points');
# 				axes[1, 1].set_ylabel('velocity(pixel/hz)');
# 				axes[1, 1].set_title('vertical velocity of flow')

# 				#axes[2, 0].plot(resultant_velocity, '*', linewidth=0.3, markersize=1)
# 				axes[2, 0].set_xlabel('feature points');
# 				axes[2, 0].set_ylabel('velocity(pixel/hz)');
# 				axes[2, 0].set_title('resultant velocity of flow')

# 				#axes[2, 1].plot(vy_flow_warp, 'x', linewidth=0.3, markersize=1)
# 				axes[2, 1].set_xlabel('feature points');
# 				axes[2, 1].set_ylabel('velocity(pixel/hz)');
# 				axes[2, 1].set_title('vertical velocity of flow in warp')

# 				figure.tight_layout()
# 				plt.show(block=True)
				#plt.wait()



				cv2.waitKey(10000)
				## Write the output

				# check if the video writer is None
				if writer is None:
					# initialize our video writer
					fourcc = cv2.VideoWriter_fourcc(*"MJPG")
					writer = cv2.VideoWriter(args["output"], fourcc, 30, (clone.shape[1], clone.shape[0]), True)
					#writer2 = cv2.VideoWriter(args["output2"], fourcc, 30, (visual), True)

					# some information on processing single frame
					if total > 0:
						elap = (end - start)
						print("[INFO] single frame took {:.4f} seconds".format(elap))
						print("[INFO] estimated total time to finish: {:.4f}".format(
							elap * total))

				# write the output frame to disk
				writer.write(clone)
				#writer2.write(visual)


