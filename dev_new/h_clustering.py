import numpy as np
import os
curdir = '/Users/Rui/Documents/Graduate/Research/ICEX:SIMI/lstm_eSelect/dev_new/'
os.chdir(curdir)
from Feature import conjoin, dist_metric
import copy

def h_clustering(Features_list,area_thres,prox_thres):
	if len(Features_list) == 0:
		return(Features_list)

	exit = False
	while exit == False:

		rmp = [] # initialize Feature removal list

		# re-order Features list by largest area to smallest
		Features_list = np.array(Features_list)
		f_areas = []
		for f in Features_list:
		  f_areas.append(f.area)

		so = np.flipud(np.argsort(f_areas))
		Features_list = Features_list[so]
		Features_list = Features_list.tolist()

		# Begin clustering
		for ff in range(len(Features_list)):
			f1 = Features_list[ff]
			if f1.parent == f1 and f1.area > area_thres: # if f1 has not been hunted and is large enough
				dist_list = []

				for f2 in Features_list:
					if f2 == f1: # if f2 is f1 or has beent hunted already, set dist to large value
						dist_list.append(1000)
					elif f2 in rmp:
						dist_list.append(1000)
					else:
						dist_list.append(dist_metric(f1,f2))
						#dist_list.append(proximity(f1,f2))

				dist = np.min(dist_list) # f1 hunts f2 thats closest to it
				if dist <= prox_thres:
					prey = Features_list[np.argmin(dist_list)]
					Features_list[ff] = copy.deepcopy(conjoin([f1,prey]))
					f1 = Features_list[ff] # reset f1 in features list
					prey.parent = f1
					rmp.append(prey)

		if not rmp: #if there are no Features to remove
			exit = True # exit

		for pp in rmp: #remove each hunted feature in rmp from Features_list
			Features_list.remove(pp)

	print('Number of clusters: ', len(Features_list))
	return(Features_list)