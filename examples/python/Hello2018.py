# Copyright(c) 2018 Senscape Corporation.
# License: Apache 2.0

# Import libs
import cv2, sys, numpy as np
sys.path.append('../../api/')
import hsapi as hs

# Load CNN to device and set scale / mean
net = hs.HS('mnist')
imgRoot = '../misc/2018_mnist/%d.jpg'

print('Hello')
for n in [1,2,3,4]:
	imgname = imgRoot % n
	img = cv2.imread(imgname)
	result = net.run(img)
	print(result[1])
