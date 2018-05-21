#! /usr/bin/env python3

# Copyright(c) 2018 Senscape Corporation.
# License: Apache 2.0

import numpy as np, cv2, sys
sys.path.append('../../api/')
import hsapi as hs

WEBCAM = False # Set to True if use Webcam

net = hs.HS('SketchGuess', zoom = True, verbose = 2)
if WEBCAM: video_capture = cv2.VideoCapture(0)

# Label paths
fcl2 = open('../misc/class_list.txt','r')
fcl = open('../misc/class_list_chn.txt','r')
class_list = fcl.readlines()
class_list_eng = fcl2.readlines()
cls = []

# Image processing params
p1 = 120
p2 = 45
ROI_ratio = 0.1

stage = 0

for line in class_list:
	cls.append(line.split(' ')[0])
	
while True:
	if WEBCAM: _, img = video_capture.read()
	else: image = net.device.GetImage(True) # Only get image

	# Image processing
	sz = image.shape
	cx = int(sz[0]/2)
	cy = int(sz[1]/2)
	ROI = int(sz[0]*ROI_ratio)
	edges = cv2.Canny(image,p1,p2)
	edges = cv2.cvtColor(edges,cv2.COLOR_GRAY2RGB)
	print(edges.shape)
	cropped = edges[cx-ROI:cx+ROI,cy-ROI:cy+ROI,:]

	kernel = np.ones((4,4),np.uint8)
	cropped = cv2.dilate(cropped,kernel,iterations = 1)
	output = net.run(cropped)[1]
	
	output_sort = np.argsort(output)[::-1]
	output_label = output_sort[:5]
	print('*******************')
	chn = ''.join([i for i in cls[stage][:-1] if not i.isdigit()])
	print('Draw %010s %s Stage:[%d]' % (chn, class_list_eng[stage], stage+1))
	print('*******************')
	cnt = 0
	for label in output_label:
		chn = ''.join([i for i in cls[label][:-1] if not i.isdigit()])
		string = '%s %s - %2.2f' % (chn,class_list_eng[label].split(' ')[0],output[label])
		print(string)
		cnt += 1
		if label == stage and output[label] > 0.1:
			print('Congratulations! Stage pass [%d]' % stage)
			stage += 1
		
	cv2.rectangle(image, (cy-ROI, cx-ROI), (cy+ROI, cx+ROI),(255,255,0), 5)
	rank = np.where(output_sort == stage)[0]
	string = '%s - Score: %2.2f' % (class_list_eng[stage].split(' ')[0:-1],1-float(rank)/345)
	
	cv2.putText(image, string, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, int(255*(1-rank/350.0)), int(255*rank/350.0)), 3)
	
	cv2.imshow('ret',image)
	cv2.imshow('ret2',cropped)
	key = cv2.waitKey(1)
	if key == ord('w'):
		p1 += 5
	elif key == ord('s'):
		p1 -= 5
	elif key == ord('e'):
		p2 += 5
	elif key == ord('d'):
		p2 -= 5
	elif key == ord('r'):
		ROI_ratio += 0.1
	elif key == ord('f'):
		ROI_ratio -= 0.1
	print([p1,p2])
