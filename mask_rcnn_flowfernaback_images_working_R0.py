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
#import matplotlib.mlab as mlab
from scipy.spatial import distance
from scipy.stats import norm
from numpy.linalg import inv
from sklearn.mixture import GaussianMixture

###------MY idea 
# Global vars:
WIDTH = 1000   # 700
STEP = 8    # 16    # this is important param to filter out moving pixel, bigger the better
QUIVER = (0, 0, 255) # (255, 100, 0)  # show in RED



def draw_flow(img, flow, step=STEP, motion_threshold=1.):
	h, w = img.shape[:2]
	y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1).astype(int)

	fx, fy = flow[y, x].T

# 	mag_new = np.sqrt(fx**2 + fy**2)
# 	# select magnitudes for those orientations
# 	cond2 = mag_new > motion_threshold
# 	temp = np.where(cond2, mag_new, 0)
#	mag_new , ang_new = cv2.cartToPolar(flow[y, x, 0].T, flow[y, x, 1].T) #mag, ang = cv2.cartToPolar(flow_roi[..., 0], flow_roi[..., 1])

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
	# warp_img = cv2.remap(img, flow_neg, None, cv2.INTER_CUBIC + cv2.INTER_LINEAR)
	return vis, warp_img, fx, fy #, mag_new #, ang_new



## calculate theta 
def cal_theta(vx, vy):
    
	fig, axs = plt.subplots(2)
	fig.suptitle('histogram showing horizontal and vertical flow of each pixel')
	axs[0].hist(vx, bins = 10)
	axs[1].hist(vy, bins = 10)
	plt.show()
    
	mean_vx = np.mean(vx) # can be used later 
	mean_vy = np.mean(vy)

    # calculate the magnitude of horizontal and vertical flow
	res_velocity = np.sqrt(vx**2 + vy**2)  ## If theta is negative it means the motion is inhibitory else excitatory
	print('res_velocity', type(res_velocity))
    
    # calculate theta 
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


	## get the actual dominant direction of flow




	
	# To get the dominant direction of motion we need to calculate 
	# the horizontal direction velocity and vertical velocity and check mean
	# get the horizontal resultant velocity
	hor_vx = res_velocity*np.cos(theta)
	hor_vx_mean = mean_vx*np.cos(theta)
	# get the horizontal resultant velocity	
	ver_vy = res_velocity*np.sin(theta)
	hor_vx_mean = mean_vx*np.cos(theta)

	## cal HOG of hor_vx, hor_vy	
# 	degrees = np.random.randint(0, 360, size=200)
# 	radians = np.deg2rad(degrees)
# 	bin_size = 20
# 	a , b=np.histogram(degrees, bins=np.arange(0, 360+bin_size, bin_size))
# 	centers = np.deg2rad(np.ediff1d(b)//2 + b[:-1])

# 	fig = plt.figure(figsize=(10,8))
# 	ax = fig.add_subplot(111, projection='polar')
# 	ax.bar(centers, a, width=np.deg2rad(bin_size), bottom=0.0, color='.8', edgecolor='k')
# 	ax.set_theta_zero_location("N")
# 	ax.set_theta_direction(-1)
# 	plt.show()
   


# 	figure, axes = plt.subplots(nrows=4, ncols=2)
# 	axes[0, 0].plot(hor_vx, '*', linewidth=0.3, markersize=1)
# 	axes[0, 0].set_xlabel('feature points');
# 	axes[0, 0].set_ylabel('velocity(pixel/hz');
# 	axes[0, 0].set_title('hor_vx velocity of flow')

# 	axes[0, 1].plot(vx, '*', linewidth=0.3, markersize=1)
# 	axes[0, 1].set_xlabel('feature points');
# 	axes[0, 1].set_ylabel('velocity(pixel/hz');
# 	axes[0, 1].set_title('vx velocity of flow')

# 	axes[1, 0].plot(mean_vx, '*', linewidth=0.3, markersize=1)
# 	axes[1, 0].set_xlabel('feature points');
# 	axes[1, 0].set_ylabel('velocity(pixel/hz');
# 	axes[1, 0].set_title('mean_vx velocity of flow')

# 	axes[1, 1].plot(hor_vx_mean, '*', linewidth=0.3, markersize=1)
# 	axes[1, 1].set_xlabel('feature points');
# 	axes[1, 1].set_ylabel('velocity(pixel/hz');
# 	axes[1, 1].set_title('hor_vx_mean velocity of flow')

# 	axes[2, 0].plot(res_velocity, '*', linewidth=0.3, markersize=1)
# 	axes[2, 0].set_xlabel('feature points');
# 	axes[2, 0].set_ylabel('velocity(pixel/hz');
# 	axes[2, 0].set_title('res_velocity of flow')


# 	axes[2, 1].plot(theta, '*', linewidth=0.3, markersize=1)
# 	axes[2, 1].set_xlabel('feature points');
# 	axes[2, 1].set_ylabel('Angle');
# 	axes[2, 1].set_title('Angular orientation of motion of the flow')

# 	axes[3, 0].plot(ver_vy, '*', linewidth=0.3, markersize=1)
# 	axes[3, 0].set_xlabel('feature points');
# 	axes[3, 0].set_ylabel('Angle');
# 	axes[3, 0].set_title('ver_vy velocity of the flow')

# 	axes[3, 1].plot(vy, '*', linewidth=0.3, markersize=1)
# 	axes[3, 1].set_xlabel('feature points');
# 	axes[3, 1].set_ylabel('Angle');
# 	axes[3, 1].set_title('vy velocity of the flow')


# 	#figure.title('Picking flow velocities and orientation')
# 	figure.tight_layout()
# 	plt.show()
# 	#print ('theta_shape', theta.shape)
	return theta, res_velocity



## Calculate Manhalanbolis distance for getting the scatteredness 
def evaluation_mat(cluster):

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

	print ('Range of cluster', R)
	print ('mean_abs_diff', mean_abs_diff)
	print ('Std_dev', std_dev)

	return R, mean_abs_diff, std_dev


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

# cam = cv2.VideoCapture('/home/abhishesh01/video_segmentation/workforJUNE2020/mask-rcnn_for_postprocessing/videos/unloading_short.mp4')
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
				
				##--------- Abhishesh START - create a black frame for a ref. as prev_frame to apply optical flow in 'segmented image'

				prev_visMask = (mask * 0).astype("uint8")
				prev_instance = cv2.bitwise_and(roi, roi, mask= prev_visMask)				
				instance = cv2.cvtColor(instance, cv2.COLOR_BGR2GRAY)
				prev_instance = cv2.bitwise_and(roi, roi, mask= prev_visMask)
				prev_instance = cv2.cvtColor(prev_instance, cv2.COLOR_BGR2GRAY)
				flow_roi = cv2.calcOpticalFlowFarneback(prev_instance, instance, flow=None, pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.5, flags=0)

				# cal flow of roi
				vis_flow_roi,vis_flow_warp, vx_flow_roi, vy_flow_roi = draw_flow(instance, flow_roi)

# 				print ('vis_flow_warp', vis_flow_warp.shape)
# 				print ('vis_flow_roi', vis_flow_roi.shape)
				# calculate warping of the next frame					
				#vis_flow_warp, vx_flow_warp, vy_flow_warp = warp_flow(vis_flow_roi, flow_roi)
				
#				print ('vx_flow_warp_mean', np.mean(vx_flow_warp))
#				print ('vy_flow_warp_mean', np.mean(vy_flow_warp))

				# cal resultant velocity of the flow vectors
				#res_vel_roi = np.sqrt((vx_flow_roi)*(vx_flow_roi) + (vy_flow_roi)*(vy_flow_roi))
				#res_vel_roi_warp = np.sqrt((vx_flow_warp)*(vx_flow_warp) + (vy_flow_warp)*(vy_flow_warp))
				
				# cal theta(for the direction of flow) of the flow using vx, vy
				flow_orientation, resultant_velocity = cal_theta(vx_flow_roi, vy_flow_roi)
                
				## cal HOOF of theta	
				bins_number = 60  # the [0, 360) interval will be subdivided into this
				# number of equal bins
				bins = np.linspace(0.0, 2 * np.pi, bins_number+1)
				n, t, _ = plt.hist(flow_orientation, bins)
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
				
				mean_ang = np.mean(flow_orientation)
				hor_vx = resultant_velocity*np.cos(flow_orientation)
				hor_vx_mean = np.mean(hor_vx)
				# get the horizontal resultant velocity	
				ver_vy = resultant_velocity*np.sin(flow_orientation)
				ver_vy_mean = np.mean(ver_vy)
				res_mag = np.sqrt(hor_vx**2 + ver_vy**2)
				diff_mean_vel = np.abs(hor_vx_mean - ver_vy_mean)

				if ver_vy_mean > 0 or ver_vy_mean < -0.001 or resultant_velocity.any() >= 2 or mean_ang > 60 or np.mean(res_mag) > 2:
					print ("process is picking")
					print ("ver_vy_mean", ver_vy_mean)
					print ("resultant_velocity", resultant_velocity.any())
					print ("mean_orientation", mean_ang)
					print ("magnitide of res_velocities", np.mean(res_mag))
                    
				else:
					print ("process is unloading")
					print ("ver_vy_mean", ver_vy_mean)
					print ("resultant_velocity", resultant_velocity.any())
					print ("mean_orientation", mean_ang)
					print ("magnitide of res_velocities", np.mean(res_mag))
					
                    


                
# 				print('D has data', len(D))
# 				## create a gaussian mixture model 
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



				#print ('flow_orientation', flow_orientation.shape)			
				#n, bins, patches = plt.hist(flow_orientation, 20000, facecolor='green', alpha=0.5)
				#plt.xlabel('angle')
				#plt.ylabel('No. of feature points')
				#plt.title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=np.mean(vel_x),\ \sigma=np.std(vel_x, axis=0)$')
				#plt.axis([-0.1, 0.1, 0, 100])
				#plt.show()
				#plt.pause(0.001)				
				
# 				# Apply K-mean clustering on res_vel_roi and flow_orientation to identify valid and unvalid values
# 				# Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
# 				criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# 				# Set flags (Just to avoid line break in the code)
# 				flags = cv2.KMEANS_RANDOM_CENTERS
# 				# Apply KMeans
# 				#compactnes,label,center = cv2.kmeans(res_vel_roi,3,None,criteria,10,flags)
# 				#compactnes,label,center = cv2.kmeans(res_vel_roi_warp,3,None,criteria,10,flags)
# 				compactnes,label,center = cv2.kmeans(flow_orientation,3,None,criteria,10,flags)
# 				compactness,labels,centers = cv2.kmeans(resultant_velocity,3,None,criteria,10,flags)
# 				
# 				#print ('labels', cv2.kmeans.labels)
# 				# Now separate the data, Note the flatten()
# 				A = resultant_velocity[labels.ravel()==0]
# 				B = resultant_velocity[labels.ravel()==1]
# 				C = resultant_velocity[labels.ravel()==2]
# 				#D = resultant_velocity[labels.ravel()==3]
# 				#E = resultant_velocity[labels.ravel()==4]

# 				At = flow_orientation[label.ravel()==0]
# 				Bt = flow_orientation[label.ravel()==1]
# 				Ct = flow_orientation[label.ravel()==2]

# 				## Calculate mean of each cluster 
# 				A_mean = np.mean(A)
# 				C_mean = np.mean(C)
# 				B_mean = np.mean(B)
# 				print ('A_mean', A.shape)
# 				print ('C_mean', C_mean)
# 				




# 				# evaluation of each cluster
# 				A_err = evaluation_mat(A)
# 				B_err = evaluation_mat(B)
# 				C_err = evaluation_mat(C)

# 				# Plot the data



# 				figure, axes = plt.subplots(nrows=3, ncols=2)
# 				axes[0, 0].plot(A, '*', linewidth=0.3, markersize=1)
# 				axes[0, 0].set_xlabel('feature points');
# 				axes[0, 0].set_ylabel('velocity(pixel/hz)');
# 				axes[0, 0].set_title('Cluster A of resultant velocity of flow')

# 				axes[0, 1].plot(At, '*', linewidth=0.3, markersize=1)
# 				axes[0, 1].set_xlabel('feature points');
# 				axes[0, 1].set_ylabel('velocity(pixel/hz)');
# 				axes[0, 1].set_title('Cluster At of angle of flow')


# 				axes[1, 0].plot(B, '*', linewidth=0.3, markersize=1)
# 				axes[1, 0].set_xlabel('feature points');
# 				axes[1, 0].set_ylabel('velocity(pixel/hz)');
# 				axes[1, 0].set_title('Cluster B of resultant velocity of flow')
# 				
# 				axes[1, 1].plot(Bt, '*', linewidth=0.3, markersize=1)
# 				axes[1, 1].set_xlabel('feature points');
# 				axes[1, 1].set_ylabel('velocity(pixel/hz)');
# 				axes[1, 1].set_title('Cluster Bt of angle of flow')


# 				axes[2, 0].plot(C, '*', linewidth=0.3, markersize=1)
# 				axes[2, 0].set_xlabel('feature points');
# 				axes[2, 0].set_ylabel('velocity(pixel/hz)');
# 				axes[2, 0].set_title('Cluster C of resultant velocity of flow')

# 				axes[2, 1].plot(Ct, '*', linewidth=0.3, markersize=1)
# 				axes[2, 1].set_xlabel('feature points');
# 				axes[2, 1].set_ylabel('velocity(pixel/hz)');
# 				axes[2, 1].set_title('Cluster Ct of angle of flow')

# 				#plt.plot(vx_flow_roi, '-', linewidth=0.3, markersize=1)
# 				#plt.scatter(centers[:,0],centers[:,0],s = 80,c = 'y', marker = 's')
# 				#plt.xlabel('X'),plt.ylabel('Y')
# 				figure.tight_layout()
# 				plt.show(block=True)


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
				
				
				###---------END ----------------

				# now, extract *only* the masked region of the ROI by passing
				# in the boolean mask array as our slice condition
				roi = roi[mask]

				####----Abhishesh START--- New idea ---

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

				### Idea ENDS--------------


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
				
# 				# create plots
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
					
# 		res_vel_list = []
# 		#for i in resultant_velocity:
# 		res_vel_list = np.concatenate((res_vel_list, resultant_velocity), axis=0)
# 	print('res_vel_list_shape', res_vel_list.size)
# 	print('res_vel_list', res_vel_list)
#			print('res_vel_list_size', np.size(res_vel_list))  

				# plotting graphs
				#n, bins, patches = plt.hist(vel_x, 10000, facecolor='green', alpha=0.75)
				#plt.xlabel('velocity')
				#plt.ylabel('No. of feature points')
				#plt.title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=np.mean(vel_x),\ \sigma=np.std(vel_x, axis=0)$')
				#plt.axis([-0.1, 0.1, 0, 100])
				#plt.show()
				#plt.pause(0.001)
				#if plt.pause:
				#	plt.waitforbuttonpress() 
