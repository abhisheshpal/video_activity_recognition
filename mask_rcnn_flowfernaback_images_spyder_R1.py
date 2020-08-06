#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 15:05:48 2020

@author: abhishesh01
"""
# new Algorithm for working on Activity classification

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
# import math
import matplotlib.pyplot as plt
#import matplotlib.mlab as mlab
# from scipy.spatial import distance
# from scipy.stats import norm
# from numpy.linalg import inv
# from sklearn.mixture import GaussianMixture
# from scipy.optimize import curve_fit
# from sklearn.cluster import KMeans
from scipy import sqrt, pi, arctan2, cos, sin
from scipy.ndimage import uniform_filter
from skimage.feature import hog
from skimage import draw
from statistics import median, mode



def hog2image(hogArray, imageSize=[32,32],orientations=9,pixels_per_cell=(8, 8),cells_per_block=(3, 3)):


    sy, sx = imageSize
    cx, cy = pixels_per_cell
    bx, by = cells_per_block

    n_cellsx = int(np.floor(sx // cx))  # number of cells in x
    n_cellsy = int(np.floor(sy // cy))  # number of cells in y

    n_blocksx = (n_cellsx - bx) + 1
    n_blocksy = (n_cellsy - by) + 1    

    hogArray = hogArray.reshape([n_blocksy, n_blocksx, by, bx, orientations])

    orientation_histogram = np.zeros((n_cellsy, n_cellsx, orientations))
    for x in range(n_blocksx):
            for y in range(n_blocksy):
                block = hogArray[y, x, :]
                orientation_histogram[y:y + by, x:x + bx, :] = block

    radius = min(cx, cy) // 2 - 1
    hog_image = np.zeros((sy, sx), dtype=float)
    for x in range(n_cellsx):
        for y in range(n_cellsy):
            for o in range(orientations):
                centre = tuple([y * cy + cy // 2, x * cx + cx // 2])
                dx = int(radius * cos(float(o) / orientations * np.pi))
                dy = int(radius * sin(float(o) / orientations * np.pi))
                # rr, cc = draw.bresenham(centre[0] - dy, centre[1] - dx,
                #                         centre[0] + dy, centre[1] + dx)
                rr, cc = draw.bresenham(centre[0] - dx, centre[1] - dy,\
                                        centre[0] + dx, centre[1] + dy)  
                hog_image[rr, cc] += orientation_histogram[y, x, o]

    return hog_image



def hof(flow, orientations=3, pixels_per_cell=(8, 8),
        cells_per_block=(3, 3), visualise=True, normalise=True, motion_threshold=2.):

    """Extract Histogram of Optical Flow (HOF) for a given image.

    Key difference between this and HOG is that flow is MxNx2 instead of MxN


    Compute a Histogram of Optical Flow (HOF) by

        1. (optional) global image normalisation
        2. computing the dense optical flow
        3. computing flow histograms
        4. normalising across blocks
        5. flattening into a feature vector

    Parameters
    ----------
    Flow : (M, N) ndarray
        Input image (x and y flow images).
    orientations : int
        Number of orientation bins.
    pixels_per_cell : 2 tuple (int, int)
        Size (in pixels) of a cell.
    cells_per_block  : 2 tuple (int,int)
        Number of cells in each block.
    visualise : bool, optional
        Also return an image of the hof.
    normalise : bool, optional
        Apply power law compression to normalise the image before
        processing.
    static_threshold : threshold for no motion

    Returns
    -------
    newarr : ndarray
        hof for the image as a 1D (flattened) array.
    hof_image : ndarray (if visualise=True)
        A visualisation of the hof image.

    References
    ----------
    * http://en.wikipedia.org/wiki/Histogram_of_oriented_gradients

    * Dalal, N and Triggs, B, Histograms of Oriented Gradients for
      Human Detection, IEEE Computer Society Conference on Computer
      Vision and Pattern Recognition 2005 San Diego, CA, USA

    """
    flow = np.atleast_2d(flow)

    """ 
    -1-
    The first stage applies an optional global image normalisation
    equalisation that is designed to reduce the influence of illumination
    effects. In practice we use gamma (power law) compression, either
    computing the square root or the log of each colour channel.
    Image texture strength is typically proportional to the local surface
    illumination so this compression helps to reduce the effects of local
    shadowing and illumination variations.
    """

    if flow.ndim < 3:
        raise ValueError("Requires dense flow in both directions")

    if normalise:
        flow = np.sqrt(flow)

    """ 
    -2-
    The second stage computes first order image gradients. These capture
    contour, silhouette and some texture information, while providing
    further resistance to illumination variations. The locally dominant
    colour channel is used, which provides colour invariance to a large
    extent. Variant methods may also include second order image derivatives,
    which act as primitive bar detectors - a useful feature for capturing,
    e.g. bar like structures in bicycles and limbs in humans.
    """

    if flow.dtype.kind == 'u':
        # convert uint image to float
        # to avoid problems with subtracting unsigned numbers in np.diff()
        flow = flow.astype('float')

    gx = np.zeros(flow.shape[:2])
    gy = np.zeros(flow.shape[:2])
    # gx[:, :-1] = np.diff(flow[:,:,1], n=1, axis=1)
    # gy[:-1, :] = np.diff(flow[:,:,0], n=1, axis=0)

    gx = flow[:,:,1]
    gy = flow[:,:,0]



    """ 
    -3-
    The third stage aims to produce an encoding that is sensitive to
    local image content while remaining resistant to small changes in
    pose or appearance. The adopted method pools gradient orientation
    information locally in the same way as the SIFT [Lowe 2004]
    feature. The image window is divided into small spatial regions,
    called "cells". For each cell we accumulate a local 1-D histogram
    of gradient or edge orientations over all the pixels in the
    cell. This combined cell-level 1-D histogram forms the basic
    "orientation histogram" representation. Each orientation histogram
    divides the gradient angle range into a fixed number of
    predetermined bins. The gradient magnitudes of the pixels in the
    cell are used to vote into the orientation histogram.
    """

    magnitude = sqrt(gx**2 + gy**2)
    orientation = arctan2(gy, gx) * (180 / pi) % 180

    sy, sx = flow.shape[:2]
    cx, cy = pixels_per_cell
    bx, by = cells_per_block

    n_cellsx = int(np.floor(sx // cx))  # number of cells in x
    n_cellsy = int(np.floor(sy // cy))  # number of cells in y

    # compute orientations integral images
    orientation_histogram = np.zeros((n_cellsy, n_cellsx, orientations))
    subsample = np.index_exp[cy // 2:cy * n_cellsy:cy, cx // 2:cx * n_cellsx:cx]
    for i in range(orientations-1):
        #create new integral image for this orientation
        # isolate orientations in this range

        temp_ori = np.where(orientation < 180 / orientations * (i + 1),
                            orientation, -1)
        temp_ori = np.where(orientation >= 180 / orientations * i,
                            temp_ori, -1)
        # select magnitudes for those orientations
        cond2 = (temp_ori > -1) * (magnitude > motion_threshold)
        temp_mag = np.where(cond2, magnitude, 0)

        temp_filt = uniform_filter(temp_mag, size=(cy, cx))
        orientation_histogram[:, :, i] = temp_filt[subsample]

    ''' Calculate the no-motion bin '''
    temp_mag = np.where(magnitude <= motion_threshold, magnitude, 0)

    temp_filt = uniform_filter(temp_mag, size=(cy, cx))
    orientation_histogram[:, :, -1] = temp_filt[subsample]

    # now for each cell, compute the histogram
    hof_image = None

    if visualise:
        from skimage import draw

        radius = min(cx, cy) // 2 - 1
        hof_image = np.zeros((sy, sx), dtype=float)
        for x in range(n_cellsx):
            for y in range(n_cellsy):
                for o in range(orientations-1):
                    centre = tuple([y * cy + cy // 2, x * cx + cx // 2])
                    dx = int(radius * cos(float(o) / orientations * np.pi))
                    dy = int(radius * sin(float(o) / orientations * np.pi))
                    rr, cc = draw.line(centre[0] - dy, centre[1] - dx,
                                            centre[0] + dy, centre[1] + dx)
                    hof_image[rr, cc] += orientation_histogram[y, x, o]

    """
    The fourth stage computes normalisation, which takes local groups of
    cells and contrast normalises their overall responses before passing
    to next stage. Normalisation introduces better invariance to illumination,
    shadowing, and edge contrast. It is performed by accumulating a measure
    of local histogram "energy" over local groups of cells that we call
    "blocks". The result is used to normalise each cell in the block.
    Typically each individual cell is shared between several blocks, but
    its normalisations are block dependent and thus different. The cell
    thus appears several times in the final output vector with different
    normalisations. This may seem redundant but it improves the performance.
    We refer to the normalised block descriptors as Histogram of Oriented
    Gradient (hog) descriptors.
    """

    n_blocksx = (n_cellsx - bx) + 1
    n_blocksy = (n_cellsy - by) + 1
    normalised_blocks = np.zeros((n_blocksy, n_blocksx,
                                  by, bx, orientations))

    for x in range(n_blocksx):
        for y in range(n_blocksy):
            block = orientation_histogram[y:y+by, x:x+bx, :]
            eps = 1e-5
            normalised_blocks[y, x, :] = block / sqrt(block.sum()**2 + eps)

    """
    The final step collects the hof descriptors from all blocks of a dense
    overlapping grid of blocks covering the detection window into a combined
    feature vector for use in the window classifier.
    """

    if visualise:
        return normalised_blocks.ravel(), hof_image
    else:
        return normalised_blocks.ravel()




###------MY idea 
# Global vars:
WIDTH = 1000   # 700
STEP = 16    # 16    # this is important param to filter out moving pixel, bigger the better
QUIVER = (0, 0, 255) # (255, 100, 0)  # show in RED


# This is for representation of optical flow in images
def draw_flow(img, flow, step=STEP):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1).astype(int)
    # transpose array
    fx, fy = flow[y, x].T
    plt.plot(fx)
    plt.show()
    # apply cartisian to polar conversion to cal mag and ang 
    mag_new , ang_new = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
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

    return vis, warp_img, fx, fy, mag_new, ang_new



## Calculate Manhalanbolis distance for getting the scatteredness 
def evaluation_mat(cluster):
	diff_mean = 0
	mean_abs_diff = 0
	diff = 0
	diff_sqr = 0
	std_dev = 0
	# create a gaussian model
	

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


	return len(cluster), mean_cluster , R, mean_abs_diff, std_dev

# calculate weighted sum of 
# TODO --


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
ap.add_argument("-c", "--confidence", type=float, default=0.5,
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
# cam = cv2.VideoCapture('/home/abhishesh01/video_segmentation/workforJUNE2020/mask-rcnn_for_postprocessing/videos/picking2.mp4')
cam = cv2.VideoCapture('/home/abhishesh01/video_segmentation/workforJUNE2020/mask-rcnn_for_postprocessing/videos/unloading1.mp4')
ret, frame = cam.read()
#prev_frame = imutils.resize(frame, width=WIDTH)
prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
prev_frame = np.float32(prev_frame)/255.0
#cv2.imshow('prev_frame', prev_frame)
print ('prev_frame_shape', np.shape(prev_frame))

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
	
	if ret == True:
	#if ret == True:
		count +=1
		print ('frame_count', count)
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
				# check to see if are going to visualize how to extract the
				# masked region itself
				if args["visualize"] > 0:
					# convert the mask from a boolean to an integer mask with
					# to values: 0 or 255, then apply the mask
					visMask = (mask * 255).astype("uint8")
					instance = cv2.bitwise_and(roi, roi, mask=visMask)
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
                
                		# flow gives flow vector vel_x and vel_y at each pixel level and its orientation
				flow_roi = cv2.calcOpticalFlowFarneback(prev_instance, instance, flow=None, pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.5, flags=0)
# 				hog_roi_picking = hog(flow_roi) #,orientations=2,pixels_per_cell=(20, 20),cells_per_block=(5, 5), visualize=True, feature_vector=True)
# 				hof_roi_picking = hof(flow_roi) #, orientations=2, pixels_per_cell=(20, 20), cells_per_block=(5, 5), visualise=True, normalise=True, motion_threshold=1.)
				hog_roi_unpicking = hog(flow_roi) #,orientations=2,pixels_per_cell=(20, 20),cells_per_block=(5, 5), visualize=True, feature_vector=True)
				hof_roi_unpicking = hof(flow_roi) #				

					# Compute magnitude of the flow (velocity) vectors and angle of 2D vector to show if activity is happening 
					# also mag and ang shows the changes in each pixel's intensity and orientation based on each pixel's flow vectors
					# whereas function draw_flow is giving the considered no. of pixels underobservation which reduces the computation of flow vector matrix
				mag, ang = cv2.cartToPolar(flow_roi[..., 0], flow_roi[..., 1])
				mag_norm = mag / np.linalg.norm(mag)
				ang_norm = ang / np.linalg.norm(ang)
				mag_avg_unpicking = np.mean(mag)
# 				mag_median_unpicking = median(mag.all())
				ang_avg_unpicking = np.mean(ang)
# 				ang_median_unpicking = median(ang.all())
				X_velocity_unpicking = flow_roi[:, :, 0]  # this is the correct way of taking out velocity in x-direction for all rows and column
				X_velocity_unpicking_avg = np.mean(X_velocity_unpicking)
				Y_velocity_unpicking = flow_roi[:, :, 1] # this is the correct way of taking out velocity in y-direction for all rows and column
				Y_velocity_unpicking_avg = np.mean(Y_velocity_unpicking)
                
# 				mag_avg_picking = np.mean(mag)
# 				ang_avg_picking = np.mean(ang)
# 				X_velocity_picking = flow_roi[0] # this 
# 				X_velocity_picking_avg = np.mean(X_velocity_picking)
# 				Y_velocity_picking = flow_roi[1]
# 				Y_velocity_picking_avg = np.mean(Y_velocity_picking)

#                 		fig = plt.figure()
# 				ax = fig.add_subplot(111)
				
# 				for m, a in zip(mag, ang):
# 					ax.scatter(m, a)
# 				plt.xlabel('length'),plt.ylabel('angle')
# 				plt.show()

				##--Apply K-means in mag and angles -----------------------
				criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
				# Set flags (Just to avoid line break in the code)
				flags = cv2.KMEANS_RANDOM_CENTERS
				z = mag	
# 				z = np.vstack((mag, ang))
				z = np.float32(z)
				# define criteria and apply kmeans()

				#Output parameters
				#compactness(or ret) : It is the sum of squared distance from each point to their corresponding centers.
				#label : This is the label array (same as 'code' in previous article) where each element marked '0', '1'.....
				#center: This is array of centers of clusters.
				# we want to create clusters of zip(mag, ang)
				ret,label,center=cv2.kmeans(z,2,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
				# Now separate the data, Note the flatten()
				C = z[label.ravel()==0]
				D = z[label.ravel()==1]
				E = z[label.ravel()==2]
				F = z[label.ravel()==3] 
# 				F = z[label.ravel()==4] 
# 				G = z[label.ravel()==5] 
# 				H = z[label.ravel()==6] 
# 				I = z[label.ravel()==7] 
# 				J = z[label.ravel()==8] 
# 				K = z[label.ravel()==9] 
				#L = z[label.ravel()==10] 
				#Plot the data
# 				plt.scatter(D[:, 0],D[:, 1],c = 'r')
				plt.scatter(C[:, 0],C[:, 1])
				plt.scatter(center[:, 0],center[:, 1], s = 80,c = 'y', marker = 's')
				plt.show()
				plt.scatter(D[:, 0],D[:, 1])
				plt.scatter(center[:, 0],center[:, 1], s = 80,c = 'y', marker = 's')
				plt.show()
				plt.scatter(E[:, 0],E[:, 1])
				plt.scatter(center[:, 0],center[:, 1], s = 80,c = 'y', marker = 's')
				plt.show()
				plt.scatter(F[:, 0],F[:, 1])
				plt.scatter(center[:, 0],center[:, 1], s = 80,c = 'y', marker = 's')
                		# plt.show()
# 				plt.scatter(G[:, 0],G[:, 1])
# 				plt.scatter(H[:, 0],H[:, 1])
# 				plt.scatter(I[:, 0],I[:, 1])
# 				plt.scatter(J[:, 0],J[:, 1])
# 				plt.scatter(K[:, 0],K[:, 1])



# 				plt.scatter(center[:, 0],center[:, 1], s = 80,c = 'y', marker = 's')
# 				plt.xlabel('LENGTH'),plt.ylabel('ANGLE')
# 				plt.show()

				##------------------------

                		# to get the magnitude values between certain angular groups
# 				fig = plt.figure()
# 				ax = fig.add_subplot(111)
# 				new_mag = []
# 				#for m, a in zip(mag, ang):
# 				if ang.any() > 0: #and ang.any() <= 6:
# 					for m, a in zip(mag, ang):
# 						new_mag.append(m) 
# 						ax.scatter(m, a)
# 				else:
# 					print('threshold yet not found')
# 				plt.xlabel('threshold_length'),plt.ylabel('threshold_angle')
# 				plt.show()

				#------------apply draw_flow method in flow of roi---------------
				vis_flow_roi,vis_flow_warp, vx_flow_roi, vy_flow_roi,  mag_roi, ang_roi = draw_flow(instance, flow_roi)
				
				create2Darray = np.vstack((vx_flow_roi, vy_flow_roi))
				create2Darray = np.float32(create2Darray)
                		
				# calculate singular value to estimate if the error is not accumulating in just one direction
				_, cal_singular, _ = np.linalg.svd(create2Darray)
				print ('cal_singular', cal_singular)

				# evaluation of each cluster
				# horizontal velocity clusters
# 				len_A, vx_cls_A_mean, R_A, mean_abs_diff_A, std_dev_A = evaluation_mat(vx_cls_A)
# 				len_B, vx_cls_B_mean, R_B, mean_abs_diff_B, std_dev_B = evaluation_mat(vx_cls_B)
# 				#len_C, vx_cls_C_mean, R_C, mean_abs_diff_C, std_dev_C = evaluation_mat(vx_cls_C)				

# 				# vertical velocity clusters
# 				len_At, vy_cls_At_mean, R_At, mean_abs_diff_At, std_dev_At = evaluation_mat(vy_cls_At)
# 				len_Bt, vy_cls_Bt_mean, R_Bt, mean_abs_diff_Bt, std_dev_Bt = evaluation_mat(vy_cls_Bt)
# 				#len_Ct, vx_cls_Ct_mean, R_Ct, mean_abs_diff_Ct, std_dev_Ct = evaluation_mat(vy_cls_Ct)
                
# 				
# 				print ('A_mean of cluster', vx_cls_A_mean)
# 				print ('A_Range of cluster', R_A)
# 				print ('A_mean_abs_diff', mean_abs_diff_A)
# 				print ('A_Std_dev', std_dev_A)

				# cal theta_res from 
				#theta_res = np.arctan(vy_cls_Bt_mean / vx_cls_B_mean)
# 				print ('theta_res', theta_res)

# 				theta_res = []
				
# 				if vx_cls_B_mean > 0 and vy_cls_Bt_mean >= 0:
#  					theta_res = np.arctan(vy_cls_Bt_mean, vx_cls_B_mean)
# 				elif vx_cls_B_mean > 0 and vy_cls_Bt_mean < 0:
#  					theta_res = np.arctan(vy_cls_Bt_mean, vx_cls_B_mean) + 2*(math.pi)
# 				elif vx_cls_B_mean < 0:
#  					theta_res = np.arctan(vy_cls_Bt_mean, vx_cls_B_mean) + np.pi
# 				elif vx_cls_B_mean == 0 and vy_cls_Bt_mean > 0:
#  					theta_res = np.pi/2
# 				elif vx_cls_B_mean == 0 and vy_cls_Bt_mean < 0:
#  					theta_res = 3*np.pi/2
# 				elif vx_cls_B_mean == 0 and vy_cls_Bt_mean == 0:
#  					theta_res = 0.
# 				print ('theta_res', theta_res)

 				
# 				print ('resultant_velocity_shape', resultant_velocity_roi.shape)
				#cv2.imshow('vis_flow_roi', vis_flow_roi)
				
				#plt.plot('flow_x', vx_flow_roi[0:1], vy_flow_roi[0:])
				#plt.show()
				#plt.plot('flow_y', vy_flow_roi[0:1], vy_flow_roi[0:])
				#plt.show()
				#cv2.imshow("Non_Segmented", prev_instance)
				#print ("Segmented", instance.shape)
				#print ("Non_Segmented", prev_instance.shape)
				prev_instance = instance				



###=============================================================================================================                    
                ####----Abhishesh START--- New idea to apply optical flow on whole frame not just on ROI ---
###=================================================================================================================

#----------------------- check if any event is happening in whole image---------------------- 

				## Once I get the mask of ROI, lets get the optical flow of the mask in each frame and from there we can catagorise activities
				# once we get the optical flow we can create another bounding box on the original frame with TAG : Picking / Not Picking

				curr_frame = copy.copy(clone)
				curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
				curr_frame = np.float32(curr_frame)/255.0
				print ('curr_frame', np.shape(curr_frame))

				#flow = cv2.calcOpticalFlowFarneback(prev_frame, curr_frame, flow=None, pyr_scale=0.2, levels=5, winsize=25, iterations=5, poly_n=7, poly_sigma=1.5, flags=0)
				flow = cv2.calcOpticalFlowFarneback(prev_frame, curr_frame, flow=None, pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.5, flags=0)

				# Compute magnite and angle of 2D vector 
				mag1, ang1 = cv2.cartToPolar(flow[..., 0], flow[..., 1])
				mag1_avg = np.mean(mag1)
                
                
				# If the magnitude is above a threshold value                 
				if mag1.any() > 0.0003:
                    			print('event is happening')
				else:
					print ('no such value found')

#                # vis mag and ang in scatter plot
				fig = plt.figure()
				ax1 = fig.add_subplot(111)
				for ma, an in zip(mag1, ang1):
 					ax1.scatter(ma, an)
				plt.xlabel('length'),plt.ylabel('angle')
				plt.show()
								
				visual, visual_warp, vel_x, vel_y = draw_flow(curr_frame, flow)
				hog_1 = hog(flow,orientations=3,pixels_per_cell=(8, 8),cells_per_block=(3, 3), visualize=True, feature_vector=True)
				hof_1 = hof(flow, orientations=3, pixels_per_cell=(8, 8), cells_per_block=(3, 3), visualise=True, normalise=False, motion_threshold=1.)
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

				# now, extract *only* the masked region of the ROI by passing
				# in the boolean mask array as our slice condition
				roi = roi[mask]

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

				figure, axes = plt.subplots(nrows=3, ncols=2)
# 				axes[0, 0].plot(resultant_velocity_roi, '*', linewidth=0.3, markersize=1)
				axes[0, 0].set_xlabel('feature points');
				axes[0, 0].set_ylabel('velocity(pixel/hz)');
				axes[0, 0].set_title('resultant velocity of flow')
				
# 				axes[0, 1].plot(flow_orientation_roi, '-', linewidth=0.3, markersize=1)
				axes[0, 1].set_xlabel('angle');
				axes[0, 1].set_ylabel('feature points)');
				axes[0, 1].set_title('resultant direction of flow')

				axes[1, 0].plot(vx_flow_roi, '-', linewidth=0.3, markersize=1)
				axes[1, 0].set_xlabel('feature points');
				axes[1, 0].set_ylabel('velocity(pixel/hz)');
				axes[1, 0].set_title('horizontal velocity of flow')

				axes[1, 1].plot(vy_flow_roi, '-', linewidth=0.3, markersize=1)
				axes[1, 1].set_xlabel('feature points');
				axes[1, 1].set_ylabel('velocity(pixel/hz)');
				axes[1, 1].set_title('vertical velocity of flow')

				#axes[2, 0].plot(resultant_velocity, '*', linewidth=0.3, markersize=1)
				axes[2, 0].set_xlabel('feature points');
				axes[2, 0].set_ylabel('velocity(pixel/hz)');
				axes[2, 0].set_title('resultant velocity of flow')

				#axes[2, 1].plot(vy_flow_warp, 'x', linewidth=0.3, markersize=1)
				axes[2, 1].set_xlabel('feature points');
				axes[2, 1].set_ylabel('velocity(pixel/hz)');
				axes[2, 1].set_title('vertical velocity of flow in warp')

				figure.tight_layout()
				plt.show(block=True)
                # plt.wait()
				plt.close(10)

				cv2.waitKey(10000)
				## Write the output

				count +=1
				print ('frame_count', count)

				# check if the video writer is None
				if writer is None:
					# initialize our video writer
					fourcc = cv2.VideoWriter_fourcc(*"MJPG")
					writer = cv2.VideoWriter(args["output"], fourcc, 30, (clone.shape[1], clone.shape[0]), True)
                    			# some information on processing single frame
					if total > 0:
						elap = (end - start)
						print("[INFO] single frame took {:.4f} seconds".format(elap))
						print("[INFO] estimated total time to finish: {:.4f}".format(elap * total))

				
				# write the output frame to disk
				writer.write(clone)
				#writer2.write(visual)
