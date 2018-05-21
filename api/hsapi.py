#! /usr/bin/env python3

# Copyright(c) 2018 Senscape Corporation.
# License: Apache 2.0

import os
import sys
import distro
import platform
import numpy
import warnings
import cv2
import pdb
from enum import Enum
from ctypes import *

# Low-level API
# x86_64, Raspberry Pi or OSX
dll_file = "libhs.so"
if distro.linux_distribution()[0] == 'Ubuntu' \
   and distro.linux_distribution()[1] == '16.04' \
   and platform.machine() == 'x86_64':
	arch_path = 'libs/linux/x86_64'
elif 'Raspbian' in distro.linux_distribution()[0] \
   and platform.machine() == 'armv7l':
	arch_path = 'libs/linux/armv7l'
elif platform.system() == 'Darwin':
	arch_path = 'libs/macos'
	dll_file = "libhs.dylib"
else:
	raise Exception("Unsupported operating system")

f = CDLL(os.path.join(os.path.dirname(__file__), arch_path, dll_file))

warnings.simplefilter('default', DeprecationWarning)


class EnumDeprecationHelper(object):
	def __init__(self, new_target, deprecated_values):
		self.new_target = new_target
		self.deprecated_values = deprecated_values

	def __call__(self, *args, **kwargs):
		return self.new_target(*args, **kwargs)

	def __getattr__(self, attr):
		if (attr in self.deprecated_values):
				warnings.warn('\033[93m' + "\"" + attr + "\" is deprecated. Please use \"" +
							  self.deprecated_values[attr] + "\"!" + '\033[0m',
							  DeprecationWarning, stacklevel=2)
				return getattr(self.new_target, self.deprecated_values[attr])
		return getattr(self.new_target, attr)


class hsStatus(Enum):
	OK = 0
	BUSY = -1
	ERROR = -2
	OUT_OF_MEMORY = -3
	DEVICE_NOT_FOUND = -4
	INVALID_PARAMETERS = -5
	TIMEOUT = -6
	NOT_FOUND = -7
	NO_DATA = -8
	GONE = -9
	UNSUPPORTED_GRAPH_FILE = -10
	MYRIAD_ERROR = -11

Status = EnumDeprecationHelper(hsStatus, {"NOTFOUND": "NOT_FOUND",
											"NODATA": "NO_DATA",
											"UNSUPPORTEDGRAPHFILE": "UNSUPPORTED_GRAPH_FILE",
											"MYRIADERROR": "MYRIAD_ERROR"})


class hsGlobalOption(Enum):
	LOG_LEVEL = 0

GlobalOption = EnumDeprecationHelper(hsGlobalOption, {"LOGLEVEL": "LOG_LEVEL"})


class hsDeviceOption(Enum):
	TEMP_LIM_LOWER = 1
	TEMP_LIM_HIGHER = 2
	BACKOFF_TIME_NORMAL = 3
	BACKOFF_TIME_HIGH = 4
	BACKOFF_TIME_CRITICAL = 5
	TEMPERATURE_DEBUG = 6
	THERMAL_STATS = 1000
	OPTIMISATION_LIST = 1001
	THERMAL_THROTTLING_LEVEL = 1002

DeviceOption = EnumDeprecationHelper(hsDeviceOption, {"THERMALSTATS": "THERMAL_STATS",
														"OPTIMISATIONLIST": "OPTIMISATION_LIST"})


class hsGraphOption(Enum):
	ITERATIONS = 0
	NETWORK_THROTTLE = 1
	DONT_BLOCK = 2
	TIME_TAKEN = 1000
	DEBUG_INFO = 1001

GraphOption = EnumDeprecationHelper(hsGraphOption, {"DONTBLOCK": "DONT_BLOCK",
													  "TIMETAKEN": "TIME_TAKEN",
													  "DEBUGINFO": "DEBUG_INFO"})


def EnumerateDevices():
	name = create_string_buffer(28)
	i = 0
	devices = []
	while True:
		if f.hsGetDeviceName(i, name, 28) != 0:
			break
		devices.append(name.value.decode("utf-8"))
		i = i + 1
	return devices


def SetGlobalOption(opt, data):
	data = c_int(data)
	status = f.hsSetGlobalOption(opt.value, pointer(data), sizeof(data))
	if status != Status.OK.value:
		raise Exception(Status(status))


def GetGlobalOption(opt):
	if opt == GlobalOption.LOG_LEVEL:
		optsize = c_uint()
		optvalue = c_uint()
		status = f.hsGetGlobalOption(opt.value, byref(optvalue), byref(optsize))
		if status != Status.OK.value:
			raise Exception(Status(status))
		return optvalue.value
	optsize = c_uint()
	optdata = POINTER(c_byte)()
	status = f.hsGetDeviceOption(0, opt.value, byref(optdata), byref(optsize))
	if status != Status.OK.value:
		raise Exception(Status(status))
	v = create_string_buffer(optsize.value)
	memmove(v, optdata, optsize.value)
	return v.raw


class Device:
	def __init__(self, name):
		self.handle = c_void_p()
		self.name = name

	def OpenDevice(self):
		status = f.hsOpenDevice(bytes(bytearray(self.name, "utf-8")), byref(self.handle))
		if status != Status.OK.value:
			raise Exception(Status(status))

	def CloseDevice(self):
		status = f.hsCloseDevice(self.handle)
		self.handle = c_void_p()
		if status != Status.OK.value:
			raise Exception(Status(status))

	def UpdateApp(self, fileName):
		status = f.hsUpdateApp(self.handle, c_char_p(fileName.encode('utf-8')))
		self.handle = c_void_p()
		if status != Status.OK.value:
			raise Exception(Status(status))

	def SetDeviceOption(self, opt, data):
		if opt == DeviceOption.TEMP_LIM_HIGHER or opt == DeviceOption.TEMP_LIM_LOWER:
			data = c_float(data)
		else:
			data = c_int(data)
		status = f.hsSetDeviceOption(self.handle, opt.value, pointer(data), sizeof(data))
		if status != Status.OK.value:
			raise Exception(Status(status))

	def GetDeviceOption(self, opt):
		if opt == DeviceOption.TEMP_LIM_HIGHER or opt == DeviceOption.TEMP_LIM_LOWER:
			optdata = c_float()
		elif (opt == DeviceOption.BACKOFF_TIME_NORMAL or opt == DeviceOption.BACKOFF_TIME_HIGH or
			  opt == DeviceOption.BACKOFF_TIME_CRITICAL or opt == DeviceOption.TEMPERATURE_DEBUG or
			  opt == DeviceOption.THERMAL_THROTTLING_LEVEL):
			optdata = c_int()
		else:
			optdata = POINTER(c_byte)()
		optsize = c_uint()
		status = f.hsGetDeviceOption(self.handle, opt.value, byref(optdata), byref(optsize))
		if status != Status.OK.value:
			raise Exception(Status(status))
		if opt == DeviceOption.TEMP_LIM_HIGHER or opt == DeviceOption.TEMP_LIM_LOWER:
			return optdata.value
		elif (opt == DeviceOption.BACKOFF_TIME_NORMAL or opt == DeviceOption.BACKOFF_TIME_HIGH or
			  opt == DeviceOption.BACKOFF_TIME_CRITICAL or opt == DeviceOption.TEMPERATURE_DEBUG or
			  opt == DeviceOption.THERMAL_THROTTLING_LEVEL):
			return optdata.value
		v = create_string_buffer(optsize.value)
		memmove(v, optdata, optsize.value)
		if opt == DeviceOption.OPTIMISATION_LIST:
			l = []
			for i in range(40):
				if v.raw[i * 50] != 0:
					ss = v.raw[i * 50:]
					end = ss.find(b'\x00')
					val = ss[0:end].decode()
					if val:
						l.append(val)
			return l
		if opt == DeviceOption.THERMAL_STATS:
			return numpy.frombuffer(v.raw, dtype=numpy.float32)
		return int.from_bytes(v.raw, byteorder='little')

	def AllocateGraph(self, graphfile,std_value=1.0,mean_value=0.0):
		hgraph = c_void_p()
		status = f.hsAllocateGraph(self.handle, byref(hgraph), graphfile, len(graphfile))
		if status != Status.OK.value:
			raise Exception(Status(status))
		return Graph(hgraph,std_value,mean_value)

	def GetImage(self,zoomMode):
		image = c_void_p()
		mode = c_bool(zoomMode)
		status = f.hsDeviceGetImage(self.handle, byref(image),mode)
		if status == Status.NO_DATA.value:
			return None, None
		if status != Status.OK.value:
			raise Exception(Status(status))

		if zoomMode == True:
			v = create_string_buffer(640*360*3)
			memmove(v, image, 640*360*3)
			image = numpy.frombuffer(v.raw, dtype=numpy.uint8).reshape(360,640,3)
		else:
			v = create_string_buffer(1920*1080*3)
			memmove(v, image, 1920*1080*3)
			image = numpy.frombuffer(v.raw, dtype=numpy.uint8)
			r = image[0:1920*1080].reshape(1080,1920)
			g = image[1920*1080:1920*1080+int(1920*1080)].reshape(1080,1920)
			b = image[1920*1080+int(1920*1080):1920*1080+int(2*1920*1080)].reshape(1080, 1920)
			image = cv2.merge(([b,g,r])).astype(numpy.uint8)

		return image

class Graph:
	def __init__(self, handle,std_value,mean_value):
		self.handle = handle
		self.userobjs = {}
		self.std_value = std_value
		self.mean_value = mean_value

	def SetGraphOption(self, opt, data):
		data = c_int(data)
		status = f.hsSetGraphOption(self.handle, opt.value, pointer(data), sizeof(data))
		if status != Status.OK.value:
			raise Exception(Status(status))

	def GetGraphOption(self, opt):
		if opt == GraphOption.ITERATIONS or opt == GraphOption.NETWORK_THROTTLE or opt == GraphOption.DONT_BLOCK:
			optdata = c_int()
		else:
			optdata = POINTER(c_byte)()
		optsize = c_uint()
		status = f.hsGetGraphOption(self.handle, opt.value, byref(optdata), byref(optsize))
		if status != Status.OK.value:
			raise Exception(Status(status))
		if opt == GraphOption.ITERATIONS or opt == GraphOption.NETWORK_THROTTLE or opt == GraphOption.DONT_BLOCK:
			return optdata.value
		v = create_string_buffer(optsize.value)
		memmove(v, optdata, optsize.value)
		if opt == GraphOption.TIME_TAKEN:
			return numpy.frombuffer(v.raw, dtype=numpy.float32)
		if opt == GraphOption.DEBUG_INFO:
			return v.raw[0:v.raw.find(0)].decode()
		return int.from_bytes(v.raw, byteorder='little')

	def DeallocateGraph(self):
		status = f.hsDeallocateGraph(self.handle)
		self.handle = 0
		if status != Status.OK.value:
			raise Exception(Status(status))

	def LoadTensor(self, tensor, userobj):
		tensor = tensor.tostring()
		userobj = py_object(userobj)
		key = c_long(addressof(userobj))
		self.userobjs[key.value] = userobj
		status = f.hsLoadTensor(self.handle, tensor, len(tensor), key)
		if status == Status.BUSY.value:
			return False
		if status != Status.OK.value:
			del self.userobjs[key.value]
			raise Exception(Status(status))
		return True

	def GetResult(self):
		tensor = c_void_p()
		tensorlen = c_uint()
		userobj = c_long()
		status = f.hsGetResult(self.handle, byref(tensor), byref(tensorlen), byref(userobj))
		if status == Status.NO_DATA.value:
			return None, None
		if status != Status.OK.value:
			raise Exception(Status(status))
		v = create_string_buffer(tensorlen.value)
		memmove(v, tensor, tensorlen.value)
		tensor = numpy.frombuffer(v.raw, dtype=numpy.float16)
		retuserobj = self.userobjs[userobj.value]
		del self.userobjs[userobj.value]
		return tensor, retuserobj.value

	def GetImage(self, zoomMode):
		image = c_void_p()
		userobj = py_object([None])
		key = c_long(addressof(userobj))
		self.userobjs[key.value] = userobj
		std_value = c_float(self.std_value)
		mean_value = c_float(self.mean_value)
		mode = c_bool(zoomMode)
		status = f.hsGetImage(self.handle, byref(image),key,std_value,mean_value,mode)
		if status == Status.NO_DATA.value:
			return None
		if status != Status.OK.value:
			del self.userobjs[key.value]
			raise Exception(Status(status))

		if zoomMode == True:
			v = create_string_buffer(640*360*3)
			memmove(v, image, 640*360*3)
			image = numpy.frombuffer(v.raw, dtype=numpy.uint8).reshape(360,640,3)
		else:
			v = create_string_buffer(1920*1080*3)
			memmove(v, image, 1920*1080*3)
			image = numpy.frombuffer(v.raw, dtype=numpy.uint8)
			r = image[0:1920*1080].reshape(1080,1920)
			g = image[1920*1080:1920*1080+int(1920*1080)].reshape(1080,1920)
			b = image[1920*1080+int(1920*1080):1920*1080+int(2*1920*1080)].reshape(1080, 1920)
			image = cv2.merge(([b,g,r])).astype(numpy.uint8)
		return image

		
# High-level API
class HS:
	def __init__(self, modelName=None, **kwargs):
		# Default
		self.verbose = 2
		self.mean = 1.0
		self.scale = 0.007843
		self.graphFolder = '../graphs/'
		self.dataFolder = '../misc/'
		self.zoom = True
		self.labels = None
		
		# Default SSD threshold
		self.threshSSD = 0.55
		
		for k,v in kwargs.items(): 
			exec('self.'+k+'=v')
		
		self.msg(' Horned Sungem ','=')
		self.devices = EnumerateDevices()
		if len(self.devices) == 0:
			print('No devices found')
			quit()
		self.device = Device(self.devices[0])
		self.device.OpenDevice()
		self.msg('Device found [0]')
		
		model_param = self.getParam(modelName)
		
		if model_param is not None:
			self.graphPath = self.graphFolder + model_param[0]
			scale = model_param[1]
			mean = model_param[2]
			self.isGray = model_param[3]
			self.netSize = model_param[4]
			self.type = model_param[5]
		else:
			self.graphPath = modelName
			self.isGray = False
			self.netSize = None
			if self.graphPath is None:
				print('Please set graph path')
				quit()
			self.type = 0
		
		try:
			self.msg(self.graphPath)
			with open(self.graphPath, mode='rb') as f:
				self.graph_byte = f.read()
				self.msg('Model loaded to Python')
		except:
			print('Error: Failed to load graph, please check file path')
			quit()
		
		try:
			self.graph = self.device.AllocateGraph(self.graph_byte, scale, -mean)
			self.msg('Model allocated to device')
		except:
			print('Error: Failed to allocate graph to device, please try to re-plug the device')
			quit()
		self.msg('','=')
		
	def run(self, img=None, **kwargs):
		if img is None:
			image = self.graph.GetImage(self.zoom)
		else:
			if self.isGray:
				image = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
			else:
				image = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
		
			img2load = cv2.resize(image,self.netSize).astype(float)
			img2load *= self.scale
			img2load -= self.mean
			self.graph.LoadTensor(img2load.astype(numpy.float16), None)

		self.imgSize = image.shape[:2]
		output, _ = self.graph.GetResult()

		for k,v in kwargs.items(): 
			exec('self.'+k+'=v')
			
		if self.type is 1 : # Classification
			output = numpy.argmax(output)
		elif self.type is 2: # SSD Face
			output = self.getBoundingBoxFromSSDResult(output, self.imgSize)
			self.labels = ['Face']
		elif self.type is 3: # SSD Obj
			output = self.getBoundingBoxFromSSDResult(output, self.imgSize)
			self.labels = ['aeroplane', 'bicycle', 'bird', 'boat',
						  'bottle', 'bus', 'car', 'cat', 'chair',
						  'cow', 'diningtable', 'dog', 'horse',
						  'motorbike', 'person', 'pottedplant',
						  'sheep', 'sofa', 'train', 'tvmonitor']
		
		# RGB -> BRG for OpenCV display
		if img is not None and not self.isGray:
			image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
			
		return [image,output]
			
	def getParam(self,modelName):
		# Model filename, scale, mean, net input is gray?, image size, result type
		if modelName is 'mnist':
			return ['graph_mnist', 0.007843, 1.0, True, (28,28), 1]
		elif modelName is 'FaceDetector':
			return ['graph_face_SSD', 0.007843, 1.0, False, (300,300), 2]
		elif modelName is 'ObjectDetector':
			return ['graph_object_SSD', 0.007843, 1.0, False, (300,300), 3]
		elif modelName is 'GoogleNet':
			return ['graph_g', 0.007843, 1.0, False, (224,224), 4]
		elif modelName is 'FaceNet':
			return ['graph_fn', 0.007843, 1.0, False, (160,160), 5]
		elif modelName is 'SketchGuess':
			return ['graph_sg', 0.007843, 1.0, False, (28,28), 6]
		else:
			self.msg('Using user\'s graph file')
			return None 

	# SSD Related:		
	def getBoundingBoxFromSSDResult(self, out_HS, size=(300,300)):
		num = int(out_HS[0])
		boxes = []
		for box_index in range(num):
			base_index = 7 + box_index * 7
			score = out_HS[base_index+2]
			if numpy.isnan(score) or score <= self.threshSSD:
				continue;
			clas = int(out_HS[base_index + 1])-1
			score = out_HS[base_index + 2]
			x1 = int(out_HS[base_index + 3] * size[1])
			y1 = int(out_HS[base_index + 4] * size[0])
			x2 = int(out_HS[base_index + 5] * size[1])
			y2 = int(out_HS[base_index + 6] * size[0])
			boxes.append([clas,score,x1,y1,x2,y2])
		return boxes
		
	def plotSSD(self, result, labels=None):
		if labels is None:
			labels = self.labels

		display_image = result[0]
		boxes = result[1]
		source_image_width = display_image.shape[1]
		source_image_height = display_image.shape[0]

		self.msg_debug('SSD [%d]: Box values' % len(boxes),'*')
		for box in boxes:
			class_id = box[0]
			percentage = int(box[1] * 100)

			label_text = self.labels[int(class_id)] + " (" + str(percentage) + "%)"
			box_w = box[4]-box[2]
			box_h = box[5]-box[3]
			if (box_w > self.imgSize[0]*0.8) or (box_h > self.imgSize[1]*0.8):
				continue	

			self.msg_debug('Box Name: %s' % self.labels[int(class_id)])
			self.msg_debug('%d %d %d %d - w:%d h:%d' %(box[2],box[3],box[4],box[5],box_w,box_h))
			
			box_color = (255, 128, 0) 
			box_thickness = 2
			cv2.rectangle(display_image, (box[2], box[3]), (box[4], box[5]), box_color, box_thickness)

			label_background_color = (255, 128, 0)
			label_text_color = (0, 255, 255)

			label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
			label_left = box[2]
			label_top = box[3] - label_size[1]
			if (label_top < 1):
				label_top = 1
			label_right = box[2] + label_size[0]
			label_bottom = box[3] + label_size[1]
			cv2.rectangle(display_image, (label_left - 1, label_top - 1), (label_right + 1, label_bottom + 1),
						  label_background_color, -1)

			cv2.putText(display_image, label_text, (label_left, label_bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_text_color, 1)
		return display_image
	
	
	# Scene Recorder Related
	def init_recorder(self):
		global annoy
		import annoy # For approximate nearest neighbour processing
		
		
		
		self.activated = False
		self.featBinLength = []
		if not hasattr(self, 'featBin'):
			self.featBin = {}
			self.featBin = {x:{} for x in ['1','2','3','4','5']}
			for n in range(1,6):
				self.featBin[str(n)]['feats'] = []
				self.featBinLength.append(0)
		# Load from file
		else:
			for n in range(1,6):
				featLen = len(self.featBin[str(n)]['feats'])
				self.featBinLength.append(featLen)
				if (featLen > 0):
					self.featDim = len(self.featBin[str(n)]['feats'][0])
			self.compressFeatBin()
			self.buildANN()
			self.dispBins()
			
			
	def record(self, result, key, **kwargs):
		self.saveFilename=self.dataFolder + 'record.dat'
		self.metric = 'euclidean'
		self.threshPerc = 0.3
		
		for k,v in kwargs.items(): 
			exec('self.'+k+'=v')
		
		if not hasattr(self, 'featBin'):
			self.featDim = result[1].shape[0]
			self.init_recorder()
			
		if key == -1:
			if self.activated:
				return self.runANN(result[1])
			return None
		key = chr(key)
			
		if key in ['1','2','3','4','5']:
			self.msg('Record to bin: ' + key)
			if key in self.featBin:
				self.featBin[key]['feats'].append(result[1])
				self.featBinLength[int(key)-1] += 1
				self.dispBins()
		elif key is 'r' or key is 'R':
			self.compressFeatBin()
			self.buildANN()
		elif key is 's' or key is 'S':
			self.saveBinsToLocal()
		elif key is 'l' or key is 'L':
			self.loadBinsToLocal()
		elif key is 'p' or key is 'P':
			self.resetBins()
		return None
			
	def compressFeatBin(self):
		binList = []
		for idx in range(5):
			if self.featBinLength[idx] > 0:
				binList.append(idx)
				
		if len(binList) > 1:
			# Use interclass distance: pick the first feature from two class and calculate a 'reference background distance'
			minDist = sys.maxsize
			for n in range(len(binList)):
				for m in range(n+1, len(binList)):
					dist = numpy.linalg.norm(self.featBin[str(binList[n]+1)]['feats'][0] - self.featBin[str(binList[m]+1)]['feats'][0])
					minDist = dist if (dist < minDist) else minDist
					
					self.msg('Compress Feature Bins','-')
					self.msg_debug('Bin[%d]-Bin[%d]:%2.2f' % (binList[n]+1, binList[m]+1, dist))
		
			self.estiBGdist = minDist
			self.thresh = minDist * self.threshPerc
			self.msg('Estimated BG dist: %2.2f' % minDist)
			self.msg('Use %2.2f as inner-dist thresh' % self.thresh)
			if self.thresh < 0.4:
				self.msg('Warning: BG dist too close!','*')
			
			self.msg('Compressing','.')

			for n in range(len(binList)):
				idx = str(binList[n]+1)
				newList = [self.featBin[idx]['feats'][0]]
				for i in range(1, self.featBinLength[binList[n]]):
					minDist = sys.maxsize
					feat2 = self.featBin[idx]['feats'][i]
					for feat in newList:
						dist = numpy.linalg.norm(feat - feat2)
						minDist = dist if (dist < minDist) else minDist
					if minDist > self.thresh:
						newList.append(feat2)
				self.featBin[idx]['feats'] = newList
				
			# Update 
			for n in range(5):
				self.featBinLength[n] = len(self.featBin[str(n+1)]['feats'])
			self.dispBins()
			self.msg('Compress finished','-')
				
		else:
			self.msg('Please record second class')
		return
		
	def buildANN(self):
		self.binList = []
		for idx in range(5):
			if self.featBinLength[idx] > 0:
				self.binList.append(idx)
				
		self.msg('Building ANN trees','-')
		for n in range(len(self.binList)):
			idx = str(self.binList[n]+1)
			self.featBin[idx]['ann'] = annoy.AnnoyIndex(self.featDim, self.metric)
			cnt = 0
			for i in range(self.featBinLength[self.binList[n]]):
				feat = self.featBin[idx]['feats'][i]
				self.featBin[idx]['ann'].add_item(cnt, feat)
				cnt += 1
			self.featBin[idx]['ann'].build(20)
			self.msg('Bin[%s] finished' % idx)
		self.msg('Building finished','-')
		self.activated = True
		
	def runANN(self,feat):
		self.msg('Running ANN','-')
		dists = []
		for n in range(5):
			idx = str(n+1)
			if 'ann' in self.featBin[idx]:
				[index, dist] = self.featBin[idx]['ann'].get_nns_by_vector(feat, 1, search_k=-1, include_distances=True)
				dists.append(-dist[0])
			else:
				dists.append(-sys.maxsize)
		
		
		result = self.softmax(numpy.array(dists))
		for n in range(5):
			self.msg_debug('[%d]: %2.2f' % (n+1, result[n]))
		
		self.msg('Probabilities','-')
		for n in range(5):
			self.msg('%s' % ('|'*int(10*result[n])))
			
		return result
		
	def saveBinsToLocal(self):
		import pickle
		with open(self.saveFilename, 'wb') as fp:
			featList = []
			for i in range(5):
				featList.append(self.featBin[str(i+1)]['feats'])
			pickle.dump(featList, fp)
		self.msg('Save complete','+')
		
	def loadBinsToLocal(self):
		import os.path
		filename = self.saveFilename
		if os.path.isfile(filename):
			import pickle
			with open(filename, 'rb') as fp:
				featList = pickle.load(fp)
			for i in range(5):
				self.featBin[str(i+1)]['feats'] = featList[i]
			self.init_recorder()
		else:
			self.msg('Cannot find data file!')
		
	def resetBins(self):
		del self.featBin
		self.msg('Reset!','+')
		
	def dispBins(self):
		self.msg('[%d]-[%d]-[%d]-[%d]-[%d]' % (self.featBinLength[0],self.featBinLength[1],self.featBinLength[2],self.featBinLength[3],self.featBinLength[4]))

			
# Util functions	
	def getImage(self):
		return self.device.GetImage(self.zoom)

	# Messager
	def msg(self, string, pad=False):
		if not pad:
			pad = ' '
	
		if self.verbose >= 1:
			print('| %s |' % string.center(30, pad))
			
	def msg_debug(self, string, pad=False):
		if not pad:
			pad = ' '
		if self.verbose >= 2:
			print('* %s *' % string.center(30, pad))
		
	# Math
	def softmax(self, x):
		e_x = numpy.exp(x - numpy.max(x))
		return e_x / e_x.sum()
		
	# Quit
	def quit(self):
		self.graph.DeallocateGraph()
		self.device.CloseDevice()
		sys.exit(1)
		return
