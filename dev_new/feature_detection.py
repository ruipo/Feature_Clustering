import numpy as np
import os
curdir = '/Users/Rui/Documents/Graduate/Research/ICEX:SIMI/lstm_eSelect/dev_new/'
os.chdir(curdir)
from Feature import Feature, conjoin

def feature_detection(S_ana_log):

	indices = np.argwhere(~np.isnan(S_ana_log)) # Find all non-NaN indices in S_ana_log
	#print(np.shape(indices))
	Features_list = [] # initialize Features_list

	for [x_ind,y_ind] in indices: # For each pair of indices (pixel)

	  if len(Features_list) == 0: # if len of feature_list is 0
	    newFeature = Feature(x_ind,y_ind) # create a new feature
	    Features_list.append(newFeature) # add new feature to feature_list

	  else:

	    border_list = [] # initialize list of logicals on where a Feature borders current pixel
	    sublist = [] # initialize list of neighboring Features that boarder current pixel
	    for currentFeature in Features_list: # for each feature in list

	      border_list.append(currentFeature.borders(x_ind,y_ind)) # find all features that boarders the pixel

	    indslist = np.where(border_list)[0] 
	    if len(indslist) == 1: # if current pixel boarders the exactly 1 Feature
	      hunterFeature = Features_list[indslist[0]] # find that Feature
	      hunterFeature.add(x_ind,y_ind) # add current pixel to that Feature

	    if len(indslist) > 1: # if the current pixel boarders more than 1 Feature
	      for ind in indslist:
	        sublist.append(Features_list[ind]) # add those Features to a list
	      for s in sublist:
	        Features_list.remove(s) # remove those Features from Features_list

	      hunterFeature = conjoin(sublist) # conjoin all Features in sublist
	      hunterFeature.add(x_ind,y_ind) # add current pixel to conjoined Feature
	      Features_list.append(hunterFeature) # add conjoined Feature to Feature_list


	    else: # if current pixel does not boarder any existing features
	      newFeature = Feature(x_ind,y_ind) # create a new feature
	      Features_list.append(newFeature) # add new feature to feature_list

	return(Features_list,indices)