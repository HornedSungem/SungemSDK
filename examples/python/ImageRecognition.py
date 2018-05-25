#! /usr/bin/env python3

# Copyright(c) 2018 Senscape Corporation.
# License: Apache 2.0

## Example of low level api

import numpy as np, cv2, sys, re
sys.path.append('../../api/')
import hsapi as hs

WEBCAM = False # Set to True if use Webcam

# Load device
devices = hs.EnumerateDevices()
dev = hs.Device(devices[0])
dev.OpenDevice()

# Load CNN model
with open('../graphs/graph_sz', mode='rb') as f:
	b = f.read()
graph = dev.AllocateGraph(b)
dim = (227,227)

# Load classes
classes=np.loadtxt('../misc/image_category.txt',str,delimiter='\t')

# Set camera mode
if WEBCAM: video_capture = cv2.VideoCapture(0)

ROI_ratio = 0.5
while True:
	if WEBCAM: _, img = video_capture.read()
	else: image_raw = dev.GetImage(True) # Only get image

	# Crop ROI
	sz = image_raw.shape
	cx = int(sz[0]/2)
	cy = int(sz[1]/2)
	ROI = int(sz[0]*ROI_ratio)
	cropped = image_raw[cx-ROI:cx+ROI,cy-ROI:cy+ROI,:]
	
	# Preprocess
	cropped = cropped.astype(np.float32)
	cropped[:,:,0] = (cropped[:,:,0] - 104)
	cropped[:,:,1] = (cropped[:,:,1] - 117)
	cropped[:,:,2] = (cropped[:,:,2] - 123)
	
	# Load data to HS
	graph.LoadTensor(cv2.resize(cropped,dim).astype(np.float16), 'user object')
	output, userobj = graph.GetResult()
	output_label = output.argsort()[::-1][:5]
	
	# Visualisation
	for i in range(5):
		label = re.search("n[0-9]+\s([^,]+)", classes[output_label[i]]).groups(1)[0]
		cv2.putText(image_raw, "%s %0.2f %%" % (label, output[output_label[i]]*100), (20, 50+i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
	cv2.rectangle(image_raw, (cy-ROI, cx-ROI), (cy+ROI, cx+ROI),(255,255,0), 5)
	cv2.imshow('Result',image_raw)
	
	key = cv2.waitKey(1)
	if key == ord('w'):
		ROI_ratio += 0.1
	elif key == ord('s'):
		ROI_ratio -= 0.1
	if ROI_ratio < 0.1:
		ROI_ratio = 0.1
