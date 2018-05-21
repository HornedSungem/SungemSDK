#! /usr/bin/env python3

# Copyright(c) 2018 Senscape Corporation.
# License: Apache 2.0

import numpy as np, cv2, sys
sys.path.append('../../api/')
import hsapi as hs

WEBCAM = False # Set to True if use Webcam
	
net = hs.HS('GoogleNet', zoom = True, verbose = 2)
if WEBCAM: video_capture = cv2.VideoCapture(0)

while True:
	if WEBCAM: _, img = video_capture.read()
	else: img = None

	# Get image descriptor
	result = net.run(img)
	key = cv2.waitKey(5)
	prob = net.record(result, key, saveFilename='../misc/record.dat')
	print(prob)
	cv2.imshow("Scene Recorder", result[0])
	cv2.waitKey(1)
