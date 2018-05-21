#! /usr/bin/env python3

# Copyright(c) 2018 Senscape Corporation.
# License: Apache 2.0

import numpy as np, cv2, sys
sys.path.append('../../api/')
import hsapi as hs

WEBCAM = False # Set to True if use Webcam
net = hs.HS('FaceDetector', zoom = True, verbose = 2, threshSSD=0.55)
if WEBCAM: video_capture = cv2.VideoCapture(0)

while True:
	if WEBCAM: _, img = video_capture.read()
	else: img = None
	result = net.run(img)
	img = net.plotSSD(result)
	cv2.imshow("Face Detector", img)
	cv2.waitKey(1)
