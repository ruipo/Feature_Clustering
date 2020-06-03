import numpy as np
import os
curdir = '/Users/Rui/Documents/Graduate/Research/ICEX:SIMI/lstm_eSelect/dev_new/'
os.chdir(curdir)
from Feature import conjoin, dist_metric, Cluster,proximity
import copy
from kneed import KneeLocator
#import random

def feature_clustering(Features_list):
	opt = False
	J = []
	if len(Features_list)<10:
		klist = np.arange(len(Features_list))+1
	else:
		klist = np.arange(10)+1
	clustered_list = []

	for k in klist:
		unclustered_list = copy.deepcopy(Features_list.tolist())
		unassigned_list = unclustered_list.copy()
		clustered_list = []

		f_areas = []
		for f in Features_list:
			f_areas.append(f.area)

		rep_list = []
		cluster_list = []
		
		# for ii in range(k):
		# 	if len(rep_list) == 0:
		# 		rep_list = random.sample(unclustered_list,1)
		# 		cluster_list.append(Cluster(rep_list[0]))
		# 		unclustered_list.remove(rep_list[0])
		# 		unassigned_list.remove(rep_list[0])
		# 	else:
		# 		unassigned_list = [ff for ff in unassigned_list if dist_metric(ff,rep_list[-1])>10]
		# 		if len(unassigned_list) == 0:
		# 			break
		# 		else:
		# 			addf = random.sample(unassigned_list,1)
		# 			rep_list.append(addf[0])
		# 			cluster_list.append(Cluster(rep_list[-1]))
		# 			unclustered_list.remove(rep_list[-1])
		# 			unassigned_list.remove(rep_list[-1])


		# rep_list = random.sample(unclustered_list,k)
		# for ii in range(k):
		# 	cluster_list.append(Cluster(rep_list[ii]))
		# 	unclustered_list.remove(rep_list[ii])

		for ii in range(k):
			if len(rep_list) == 0:
				ind = np.argmax(f_areas)
				#unclustered_list.append(Features_list[ind])
				rep_list.append(unclustered_list[ind])
				cluster_list.append(Cluster(unclustered_list[ind]))
				f_areas.remove(f_areas[ind])
				unassigned_list.remove(unclustered_list[ind])
				unclustered_list.remove(unclustered_list[ind])
				
			else:
				f_areas = []
				unassigned_list = [ff for ff in unassigned_list if proximity(ff,rep_list[-1])>0]
				for ff in unassigned_list:
						f_areas.append(ff.area)

				if len(f_areas) == 0:
					break
				else:
					ind = np.argmax(f_areas)
					#unclustered_list.append(Features_list[ind])
					rep_list.append(unassigned_list[ind])
					cluster_list.append(Cluster(unassigned_list[ind]))
					unclustered_list.remove(unassigned_list[ind])
					unassigned_list.remove(unassigned_list[ind])

		print('Trying ', len(rep_list), ' Clusters...')
		#print(len(unclustered_list))

		cost_dist_list = []
		it = 0
		lim = 5
		epi = 0
		while it < lim:
			#print(it)
			cost_dist = 0

			for f1 in unclustered_list:
				dist_list = []
				
				for f2 in rep_list:
					dist = dist_metric(f1,f2)
					dist_list.append(dist)

				minind = np.argmin(dist_list)

				cluster_list[minind].add(f1)
				cost_dist = cost_dist+dist_list[minind]

			if it < lim-1:
				# for ff in rep_list:
				# 	unclustered_list.append(ff)

				for cc in range(len(cluster_list)):
					prev_rep = rep_list[cc]
					unclustered_list.append(prev_rep)
					new_rep = cluster_list[cc].setrep(epi)
					rep_list[cc] = new_rep
					unclustered_list.remove(new_rep)
					cluster_list[cc].clear()
					cluster_list[cc].rep = rep_list[cc]
					cluster_list[cc].add(rep_list[cc])
				#print(rep_list)

			cost_dist_list.append(cost_dist)
			it = it + 1
			#epi = 0.75*epi

		for cc in range(len(cluster_list)):
				clustered_list.append(conjoin(cluster_list[cc].content))

		print(cost_dist)

		J.append(cost_dist)

		# if len(J)>1 and 0<=J[-2]-J[-1]<20:
		# 	k_opt = k-1
		# 	opt = True
		# 	break

	if opt == False:
		if len(J) < 2:
			k_opt = 1
			print(J)
			print('number of clusters: ',k_opt)
		else:
			# Jgrad = np.gradient(J)
			# k_opt = np.argmin(np.abs(Jgrad[np.abs(Jgrad)>0]))+1
			# if k_opt == 0:
			# 	k_opt = 1

			k_opt = KneeLocator(np.arange(len(J))+1, J, curve='convex', direction='decreasing')
			if k_opt.knee==None:
				k_opt = k
			else:
				k_opt = k_opt.knee+1

			if J == 0:
				k_opt = k_opt-1
			print(J)
			print('number of clusters: ',k_opt)

	clustered_list = []

	k = k_opt

	unclustered_list = copy.deepcopy(Features_list.tolist())
	unassigned_list = unclustered_list.copy()
	clustered_list = []

	f_areas = []
	for f in Features_list:
		f_areas.append(f.area)

	rep_list = []
	cluster_list = []

	# for ii in range(k):
	# 	if len(rep_list) == 0:
	# 		rep_list = random.sample(unclustered_list,1)
	# 		cluster_list.append(Cluster(rep_list[0]))
	# 		unclustered_list.remove(rep_list[0])
	# 		unassigned_list.remove(rep_list[0])
	# 	else:
	# 		unassigned_list = [ff for ff in unassigned_list if dist_metric(ff,rep_list[-1])>10]
	# 		if len(unassigned_list) == 0:
	# 			break
	# 		else:
	# 			addf = random.sample(unassigned_list,1)
	# 			rep_list.append(addf[0])
	# 			cluster_list.append(Cluster(rep_list[-1]))
	# 			unclustered_list.remove(rep_list[-1])
	# 			unassigned_list.remove(rep_list[-1])

	# rep_list = random.sample(unclustered_list,k)
	# for ii in range(k):
	# 	cluster_list.append(Cluster(rep_list[ii]))
	# 	unclustered_list.remove(rep_list[ii])

	for ii in range(k):
		if len(rep_list) == 0:
			ind = np.argmax(f_areas)
			#unclustered_list.append(Features_list[ind])
			rep_list.append(unclustered_list[ind])
			cluster_list.append(Cluster(unclustered_list[ind]))
			f_areas.remove(f_areas[ind])
			unassigned_list.remove(unclustered_list[ind])
			unclustered_list.remove(unclustered_list[ind])
			
		else:
			f_areas = []
			unassigned_list = [ff for ff in unassigned_list if proximity(ff,rep_list[-1])>0]
			for ff in unassigned_list:
					f_areas.append(ff.area)

			if len(f_areas) == 0:
				break
			else:
				ind = np.argmax(f_areas)
				#unclustered_list.append(Features_list[ind])
				rep_list.append(unassigned_list[ind])
				cluster_list.append(Cluster(unassigned_list[ind]))
				unclustered_list.remove(unassigned_list[ind])
				unassigned_list.remove(unassigned_list[ind])

	cost_dist_list = []
	it = 0
	lim = 5
	epi = 0
	while it < lim:
		#print(it)
		cost_dist = 0

		for f1 in unclustered_list:
			dist_list = []
			
			for f2 in rep_list:
				dist = dist_metric(f1,f2)
				dist_list.append(dist)

			minind = np.argmin(dist_list)

			cluster_list[minind].add(f1)
			cost_dist = cost_dist+dist_list[minind]

		if it < lim-1:
			for cc in range(len(cluster_list)):
				prev_rep = rep_list[cc]
				unclustered_list.append(prev_rep)
				new_rep = cluster_list[cc].setrep(epi)
				rep_list[cc] = new_rep
				unclustered_list.remove(new_rep)
				cluster_list[cc].clear()
				cluster_list[cc].rep = rep_list[cc]
				cluster_list[cc].add(rep_list[cc])
			#print(rep_list)
			#print(rep_list)

		cost_dist_list.append(cost_dist)
		it = it + 1
		#epi = 0.75*epi

	for cc in range(len(cluster_list)):
			clustered_list.append(conjoin(cluster_list[cc].content))

	return(clustered_list)


