import numpy as np
import matplotlib.dates as mds
import random

class Feature():

	def __init__(self,x,y):
		self.area = 1
		self.pixels=[(x,y)]
		self.orientation = 0 # 1 for vertical, 0 for square-ish, -1 for horizontal
		self.parent = self
		self.elevmax = -1
		self.start_f = -1 # Feature Start Freq in Hz
		self.end_f = -1 # Feature End Freq in Hz
		self.start_t = -1 # Feature Start time in epoch time
		self.end_t = -1 # Feature End time in epoch time
		self.type = ''


	def contains(self,xpos,ypos):
		if ((xpos,ypos) in self.pixels):
			return(True)
		else:
			return(False)

	def add(self,xpos,ypos):
		if ((xpos,ypos) in self.pixels):
			print('pixel already in feature!')
		else:
			self.pixels.append((xpos,ypos))
			self.area = self.area+1

	def remove(self,xpos,ypos):
		if ((xpos,ypos) in self.pixels):
			self.pixels.remove((xpos,ypos))
			self.area = self.area-1
		else:
			print('pixel not in feature!')


	def borders(self,xpos,ypos):
		if ((xpos-1,ypos+1) in self.pixels) or ((xpos-1,ypos) in self.pixels)\
		or ((xpos-1,ypos-1) in self.pixels) or ((xpos,ypos-1) in self.pixels)\
		or ((xpos,ypos+1) in self.pixels) or ((xpos+1,ypos-1) in self.pixels)\
		or ((xpos+1,ypos) in self.pixels) or ((xpos+1,ypos+1) in self.pixels):
			return(True)
		else:
			return(False)

	def hunt(self,obj):
		for p in obj.pixels:
			if p in self.pixels:
				continue
			else:
				self.add(p[0],p[1])

	def stats(self,flist,tlist):
		temp = np.array(self.pixels)

		max_f = flist[max(temp[:,0])]
		min_f = flist[min(temp[:,0])]
		mean_f = flist[int(np.mean(temp[:,0]))]
		self.start_f = min_f
		self.end_f = max_f
		diff_f = max_f-min_f
 		
		max_t = mds.num2epoch(tlist[max(temp[:,1])])
		min_t = mds.num2epoch(tlist[min(temp[:,1])])
		mean_t = mds.num2epoch(tlist[int(np.mean(temp[:,1]))])
		self.start_t = min_t
		self.end_t = max_t
		diff_t = max_t-min_t

		return(max_t, min_t, mean_t,max_f,min_f,mean_f)


def conjoin(list):
	arealist = []

	for f in list:
		arealist.append(f.area)

	hunter_ind = np.argmax(arealist)
	hunter = list[hunter_ind]

	for f in list:

		if f == hunter:
			continue

		else:
			hunter.hunt(f)

	return(hunter)

def slope(obj):
	pixels = np.array(obj.pixels)

	slope =(np.max(pixels[:,0])-np.min(pixels[:,0]))/(np.max(pixels[:,1])-np.min(pixels[:,1])+0.01)

	if slope >=1.2:
		obj.orientation = 1
	elif slope <=0.8:
		obj.orientation = -1
	else:
		obj.orientation = 0

	return(slope)


def proximity(obj1, obj2):
	pixels1 = np.array(obj1.pixels)
	pixels2 = np.array(obj2.pixels)

	prox = []
	if pixels1.shape[0] <= pixels2.shape[0]:
		for p in pixels1:
			prox.append(np.min(np.sum((pixels2 - p)**2, axis=1)))
	else:
		for p in pixels2:
			prox.append(np.min(np.sum((pixels1 - p)**2, axis=1)))

	min_prox = np.sqrt(np.min(prox))

	return(min_prox)

def prox_y(obj1,obj2):
	pixels1 = np.array(obj1.pixels)
	pixels2 = np.array(obj2.pixels)
	prox_y = []
	if pixels1.shape[0] <= pixels2.shape[0]:
		for p in pixels1:
			prox_y.append(np.min(np.abs(pixels2[:,1] - p[1])))
	else:
		for p in pixels2:
			prox_y.append(np.min(np.abs(pixels1[:,1] - p[1])))
	return(np.min(prox_y))

def prox_x(obj1,obj2):
	pixels1 = np.array(obj1.pixels)
	pixels2 = np.array(obj2.pixels)
	prox_x = []
	if pixels1.shape[0] <= pixels2.shape[0]:
		for p in pixels1:
			prox_x.append(np.min(np.abs(pixels2[:,0] - p[0])))
	else:
		for p in pixels2:
			prox_x.append(np.min(np.abs(pixels1[:,0] - p[0])))
	return(np.min(prox_x))



def dist_metric(obj1, obj2):
	pdist = proximity(obj1,obj2)
	pxdist = prox_x(obj1,obj2)
	pydist = prox_y(obj1,obj2)
	s1 = slope(obj1)
	s2 = slope(obj2)

	if (obj1.orientation == 1 and obj2.orientation == 1 and obj1.area>30 and obj2.area>30):
	# if objs 1 and 2 both have vertical orientation and are both large
		if pydist>5:
			min_dist = 500
		else:
			min_dist = 0.1*pdist+0.9*pydist

	elif (obj1.orientation == -1 and obj2.orientation == -1 and obj1.area>30 and obj2.area>30):
	# if objs 1 and 2 both have horizontal orientation and are both large
		if pxdist>5:
			min_dist = 500
		else:
			min_dist = 0.2*pdist+0.8*pxdist

	elif (obj1.orientation != obj2.orientation and obj1.orientation != 0 and obj2.orientation != 0 and obj1.area>25 and obj2.area>25):
	# if objs 1 and 2 have different orientation (one vertical, one horizontal) and are both large
		min_dist = 500

	elif (obj1.area<20 and obj2.area<20): # if objs 1 and 2 are both small
		min_dist = 0.3*pdist+0.7*np.min([pydist,pxdist])

	else: #else
		min_dist = 0.4*pdist+0.6*pydist

	return(min_dist)




# def dist_metric(obj1, obj2):
# 	pdist = proximity(obj1,obj2)
# 	pixels1 = np.array(obj1.pixels)
# 	pixels2 = np.array(obj2.pixels)

# 	#prox_x = []
# 	prox_y = []
# 	if pixels1.shape[0] <= pixels2.shape[0]:
# 		for p in pixels1:
# 			#prox_x.append(np.min(np.abs(pixels2[:,0] - p[0])))
# 			prox_y.append(np.min(np.abs(pixels2[:,1] - p[1])))
# 	else:
# 		for p in pixels2:
# 			#prox_x.append(np.min(np.abs(pixels1[:,0] - p[0])))
# 			prox_y.append(np.min(np.abs(pixels1[:,1] - p[1])))

# 	#min_dist = np.mean([pdist, np.min(prox_y)])
# 	min_dist = 0.5*pdist+0.5*np.min(prox_y)

# 	return(min_dist)

# def dist_metric(obj1, obj2):
# 	pixels1 = np.array(obj1.pixels)
# 	pixels2 = np.array(obj2.pixels)

# 	min_dist = 0
# 	if pixels1.shape[0] <= pixels2.shape[0]:
# 		for p in pixels1:
# 			min_dist_list = []
# 			prox_x = np.abs(pixels2[:,0] - p[0])
# 			prox_y = np.abs(pixels2[:,1] - p[1])

# 			for ii in range(len(prox_x)):
# 				min_dist_list.append(np.min([prox_x[ii],prox_y[ii]]))

# 			min_dist = min_dist + np.mean(min_dist_list)

# 		min_dist = min_dist/pixels1.shape[0]

# 	else:
# 		for p in pixels2:
# 			min_dist_list = []
# 			prox_x = np.abs(pixels1[:,0] - p[0])
# 			prox_y = np.abs(pixels1[:,1] - p[1])

# 			for ii in range(len(prox_x)):
# 				min_dist_list.append(np.min([prox_x[ii],prox_y[ii]]))

# 			min_dist = min_dist + np.mean(min_dist_list)

# 		min_dist = min_dist/pixels2.shape[0]

# 	return(min_dist)


class Cluster():

	def __init__(self,obj1):
		self.rep = obj1
		self.size = 1
		self.content = [obj1]

	def add(self,obj1):
		self.content.append(obj1)

	def clear(self):
		self.content = []

	def setrep(self,epsilon=0):
		if len(self.content) == 1:
			self.rep = self.content[0]
		else:
			rnum = random.uniform(0,1)

			if rnum >= epsilon:
				dist_list1 = []
				for f1 in self.content:
					dist = 0
					for f2 in self.content:
						dist = dist + dist_metric(f1,f2)
					dist_list1.append(dist)

				ind = np.argmin(dist_list1)
				self.rep = self.content[ind]

			else:
				self.rep = random.sample(self.content,1)[0]

		return(self.rep)







