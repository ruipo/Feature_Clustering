############################### Imports ##################################################
import os
curdir = '/Users/Rui/Documents/Graduate/Research/Feature_Clustering/sidex20/'
os.chdir(curdir)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mds
import librosa
import librosa.display
import cv2
import copy
from sidex_load import sidex_readin
from Feature import conjoin
from feature_detection import feature_detection
from h_clustering import h_clustering
import convex_hull
from datetime import datetime

from save_obj import save_object

path = '/Users/Rui/Documents/Graduate/Research/SIDEX/SIDEX20/sidex20_calibration/'
FS = 1000
NUM_CHANNELS = 16

n_fft = 256
hop_length = int(n_fft/2)
fmin = 0
fmax = 96 # max frequency to examine
n_mels = 32 #
mask_thres = 50 # noise level threshold to set 0-1 mask
area_thres = 25 # min size of Feature Area to save a Feature
prox_thres = 3
num_analysis_file = 1

###################################### Get Noise Sample ##################################################
print('Calculating Background Noise ...')

first_file_noise = 0
last_file_noise = 10

noise_in,time_data,ffname = sidex_readin(path,FS,NUM_CHANNELS,first_file_noise,last_file_noise) # read in stat and ana files centered around first_file
num_nfft_noise = int(np.ceil(np.shape(noise_in)[0]/hop_length)) # calculate number of total nfft bins for stats data
S_noise = np.zeros((n_mels,num_nfft_noise,NUM_CHANNELS)) # initialize stats data spectrogram matrix

for c in range(12):
  #print(c) # for each channel, determine stats data mel-spectrogram
  S_noise[:,:,c] = librosa.feature.melspectrogram(y=np.array(noise_in[:,c]), sr=FS, n_fft=n_fft, hop_length=hop_length,fmax=fmax,n_mels=n_mels)

# S_noise_start = np.mean(S_noise,axis=2) # Average over all channels
# f_means_start = np.mean(S_noise,axis=1) # get mean of each f bin in stats data spectrogram
# #f_vars = np.std(S_noise,axis=1) # get variance of each f bin in stats data spectrogram

S_noise = np.mean(S_noise,axis=2) # Average over all channels
f_means = np.mean(S_noise,axis=1)

###################################### Get data Sample ##################################################
first_file = 0
last_file = 90

while first_file < last_file:
	print('Working on file: ', first_file)
	print('Getting data ... ')

	data,time,ffname = sidex_readin(path,FS,NUM_CHANNELS,first_file,first_file+num_analysis_file)

	utc_starttime = datetime.strptime(ffname[-26:-4], "%Y%m%dT%H%M%Sp%f")
	t_ep_start = (utc_starttime - datetime(1970, 1, 1)).total_seconds()+(5*60*60)
	# time = t_ep_start+time
	# tcoords = mds.epoch2num(time)

	# t_ep_start = 1579980698.215000
	# time = t_ep_start+time
	# tcoords = mds.epoch2num(time)

	# tcalib = np.loadtxt(curdir+'calib_event_eptimes.txt')
	# tcalib_coords = mds.epoch2num(tcalib)

	# plt.plot(tcoords,data[:,0])
	# plt.plot(tcalib_coords,np.zeros(tcalib_coords.shape[0]),'r*')
	# ax = plt.gca()
	# ax.xaxis.set_major_formatter(formatter=mds.DateFormatter('%H:%M:%S'))
	# plt.grid()
	# plt.show()

	num_nfft_tot = int(np.ceil(np.shape(data)[0]/hop_length)) # calculate number of total nfft bins for stats data
	flist = librosa.core.mel_frequencies(n_mels=n_mels, fmin=fmin, fmax=fmax, htk=False)
	tlist = (1/FS)*np.linspace(0,np.shape(data)[0],num_nfft_tot) # get list of times for spectrogram x-axis
	tcoords = mds.epoch2num(tlist+t_ep_start)
	S = np.zeros((n_mels,num_nfft_tot,NUM_CHANNELS)) # initialize stats data spectrogram matrix

	for c in range(12): # for each channel, determine stats data mel-spectrogram
	  S[:,:,c] = librosa.feature.melspectrogram(y=np.array(data[:,c]), sr=FS, n_fft=n_fft, hop_length=hop_length,fmax=fmax,n_mels=n_mels)
	S = np.mean(S,axis=2)

	#S = cv2.GaussianBlur(S,ksize=(3,3),sigmaX=1,sigmaY=1) 
	S_ana_f = copy.deepcopy(S)

	for row in range(S.shape[0]): # normalize each frequency bin to zero mean and unit variance
		S_ana_f[row,:] = (S[row,:]/f_means[row])#/f_vars[row] 

	S_ana_gradx = cv2.Sobel(S_ana_f,cv2.CV_64F,1,0,ksize=5)
	S_ana_grady = cv2.Sobel(S_ana_f,cv2.CV_64F,0,1,ksize=5)
	S_ana_gradl = np.maximum(np.abs(S_ana_gradx),np.abs(S_ana_grady))
	S_ana_gradl = cv2.morphologyEx(S_ana_gradl, cv2.MORPH_OPEN, np.ones((3, 3)))
	close_mask = cv2.morphologyEx(S_ana_gradl, cv2.MORPH_CLOSE, np.ones((3, 3)))

	close_mask[close_mask<=mask_thres] = 0
	close_mask[close_mask!=0] = 1

	S_ana_log = copy.deepcopy(S_ana_f) # calculate log value of ana spectrogram
	S_ana_log = S_ana_log*close_mask
	S_ana_log[S_ana_log <= 0] = float('nan') # set all values in S_ana_log that are < db_thres to NaN


	######################################## Feature Detection ################################################################
	print('Detecting Features ... ')

	Features_list,indices = feature_detection(S_ana_log)

	############################### Group Features ###############################################
	print('Grouping Features ... ')

	Features_list = h_clustering(Features_list,area_thres,prox_thres)

	################################### Get Saved Feature Stats and Plot ############################################
	print('Generating/Saving Plot ... ')
	S_noise_cur = copy.deepcopy(S)

	Features_list_copy = copy.deepcopy(Features_list) # get a copy of Features_list
	S_g_log_test = copy.deepcopy(S_ana_log) # get a copy of S_ana_log
	filtered_Features_list = [] # initialize a list to hold all Features to be saved
	Features_stats = [] # initialize matrix to record stats of saved Features
	hull_list = [] # initilized list to record hull of saved Features

	# remove all non-Features from plot
	if (np.any(Features_list_copy)):
	  big_Feature = conjoin(Features_list_copy) 
	  for [x_ind,y_ind] in indices:
	  	if (x_ind,y_ind) not in big_Feature.pixels:
	  		S_g_log_test[(x_ind,y_ind)] = np.float('nan')

	for ent in Features_list: # For each Feature
		ent.stats(flist,tcoords)
		# # print(ent.area)
		# # print((ent.end_f-ent.start_f)/(ent.end_t-ent.start_t))

		if (ent.area <= area_thres):# or ((ent.end_f-ent.start_f)/(ent.end_t-ent.start_t)>=100)): # if Feature area <= area_thres or if feature is stbb, set Feature pixels to NaN
			for p in ent.pixels:
				S_g_log_test[p] = np.float('nan')
		else: # else, keep the Feature, add to fFlist, determine its stats, calculate the hull of the Feature
			for p in ent.pixels:
				S_noise_cur[p] = np.float('nan')

			if (ent.end_t-ent.start_t <=2): #if feature is st, set Feature type to st, else set to wc
				ent.type = 'st'
			else:
				ent.type = 'lt' 

			filtered_Features_list.append(ent)
			Features_stats.append(ent.stats(flist,tcoords))
			save_object(ent,curdir+'Features/'+str(ent.start_t)+'.pkl')
			hull = convex_hull.convex_hull(np.array(ent.pixels).T)
			hull = np.vstack((hull, hull[0]))
			hull_list.append(hull)

	print('Number of saved clusters: ', len(filtered_Features_list))
	# Set noise level for next iteration
	f_means_cur = np.nanmean(S_noise_cur,axis=1)
	f_means = 0.5*f_means+0.5*f_means_cur

	fig = plt.figure(figsize=(20,8))

	ax1 = plt.subplot(1,3,1,autoscale_on=True)
	librosa.display.specshow(10*np.log10(S),x_coords=tcoords, y_coords=flist,x_axis='time',y_axis='mel', sr=FS, fmax=fmax)
	cax = plt.colorbar()
	cax.set_label('dB',labelpad=-30, y=1.05, rotation=0,fontsize=20,size='large')
	ax = plt.gca()
	ax.xaxis.set_major_formatter(formatter=mds.DateFormatter('%H:%M:%S'))
	plt.xticks(fontsize=20)
	plt.yticks(np.arange(fmin, fmax, 16),fontsize=20)
	# plt.xlabel('Time')
	# plt.ylabel('Frequency (Hz)')
	# plt.title('Conventional')
	ax1.set_xlabel('Time',fontsize=20)
	ax1.set_ylabel('Frequency (Hz)',fontsize=20)
	ax1.set_title('Conventional',fontsize=20)
	plt.clim(-80,-50)
	plt.set_cmap('magma')

	# ax2 = plt.subplot(2,4,2,sharey=ax1,autoscale_on=True)
	# librosa.display.specshow(S_ana_f,x_coords=tcoords, y_coords=flist,x_axis='time',y_axis='mel', sr=FS, fmax=fmax)
	# cax = plt.colorbar()
	# cax.set_label(' ',labelpad=-30, y=1.05, rotation=0)
	# ax = plt.gca()
	# ax.xaxis.set_major_formatter(formatter=mds.DateFormatter('%H:%M:%S'))
	# plt.xticks(fontsize=5)
	# plt.yticks(np.arange(fmin, fmax, 96),fontsize=5)
	# #plt.clim(0,0.000005)
	# plt.xlabel('')
	# plt.ylabel('')
	# plt.title('F-Normalized')

	# ax3 = plt.subplot(2,4,3,sharey=ax1,autoscale_on=True)
	# librosa.display.specshow(np.abs(S_ana_gradx),x_coords=tcoords, y_coords=flist,x_axis='time',y_axis='mel', sr=FS, fmax=fmax)
	# cax = plt.colorbar()
	# cax.set_label(' ',labelpad=-30, y=1.05, rotation=0)
	# ax = plt.gca()
	# ax.xaxis.set_major_formatter(formatter=mds.DateFormatter('%H:%M:%S'))
	# plt.xticks(fontsize=5)
	# plt.yticks(np.arange(fmin, fmax, 96),fontsize=5)
	# #plt.clim(0,0.000005)
	# plt.xlabel('')
	# plt.ylabel('')
	# plt.title('xGradient')

	# ax4 = plt.subplot(2,4,4,sharey=ax1,autoscale_on=True)
	# librosa.display.specshow(np.abs(S_ana_grady),x_coords=tcoords, y_coords=flist,x_axis='time',y_axis='mel', sr=FS, fmax=fmax)
	# cax = plt.colorbar()
	# cax.set_label(' ',labelpad=-30, y=1.05, rotation=0)
	# ax = plt.gca()
	# ax.xaxis.set_major_formatter(formatter=mds.DateFormatter('%H:%M:%S'))
	# plt.xticks(fontsize=5)
	# plt.yticks(np.arange(fmin, fmax, 96),fontsize=5)
	# #plt.clim(0,50)
	# plt.xlabel('')
	# plt.ylabel('')
	# plt.title('yGradient')

	# ax5 = plt.subplot(2,4,5,sharey=ax1,autoscale_on=True)
	# librosa.display.specshow(S_ana_gradl,x_coords=tcoords, y_coords=flist,x_axis='time',y_axis='mel', sr=FS, fmax=fmax)
	# cax = plt.colorbar()
	# cax.set_label(' ',labelpad=-30, y=1.05, rotation=0)
	# ax = plt.gca()
	# ax.xaxis.set_major_formatter(formatter=mds.DateFormatter('%H:%M:%S'))
	# plt.xticks(fontsize=5)
	# plt.yticks(np.arange(fmin, fmax, 96),fontsize=5)
	# #plt.clim(0,10)
	# plt.xlabel('')
	# plt.ylabel('Frequency (Hz)')
	# plt.title('LGradient')

	# ax6 = plt.subplot(2,4,6,sharey=ax1,autoscale_on=True)
	# librosa.display.specshow(close_mask,x_coords=tcoords, y_coords=flist,x_axis='time',y_axis='mel', sr=FS, fmax=fmax)
	# cax = plt.colorbar()
	# cax.set_label(' ',labelpad=-30, y=1.05, rotation=0)
	# ax = plt.gca()
	# ax.xaxis.set_major_formatter(formatter=mds.DateFormatter('%H:%M:%S'))
	# plt.xticks(fontsize=5)
	# plt.yticks(np.arange(fmin, fmax, 96),fontsize=5)
	# #plt.clim(0,1)
	# plt.xlabel('')
	# plt.ylabel('')
	# plt.title('mask')

	ax2 = plt.subplot(1,3,2,sharey=ax1,autoscale_on=True)
	if np.where(~np.isnan(S_g_log_test))[0].shape[0]>0:
		librosa.display.specshow(S_ana_log,x_coords=tcoords, y_coords=flist,x_axis='time',y_axis='mel', sr=FS, fmax=fmax)
		cax = plt.colorbar()
		cax.set_label(' ',labelpad=-30, y=1.05, rotation=0,fontsize=20,size='large')
		ax = plt.gca()
		ax.xaxis.set_major_formatter(formatter=mds.DateFormatter('%H:%M:%S'))
		plt.xticks(fontsize=20)
		plt.yticks(np.arange(fmin, fmax, 16),fontsize=20)
		ax2.set_xlabel('Time',fontsize=20)
		ax2.set_title('Post-Processing',fontsize=20)
		plt.clim(0,10)
		plt.ylabel('')
		plt.grid(True)

	ax3 = plt.subplot(1,3,3,sharey=ax1,autoscale_on=True)
	if np.where(~np.isnan(S_g_log_test))[0].shape[0]>0:
	     
	  librosa.display.specshow(S_g_log_test,x_coords=tcoords, y_coords=flist,x_axis='time', y_axis='mel', sr=FS, fmax=fmax)
	  cax = plt.colorbar()
	  cax.set_label('',labelpad=-30, y=1.05, rotation=0,fontsize=20,size='large')
	  ax = plt.gca()
	  ax.xaxis.set_major_formatter(formatter=mds.DateFormatter('%H:%M:%S'))
	  plt.xticks(fontsize=20)
	  plt.yticks(np.arange(fmin, fmax, 16),fontsize=20)
	  plt.clim(0,10)
	  #plt.xlabel('Time')
	  plt.ylabel('')
	  #plt.title('Post-Clustering')
	  ax3.set_xlabel('Time',fontsize=20)
	  #ax3.set_ylabel('Frequency (Hz)',fontsize=20)
	  ax3.set_title('Post-Clustering & Type Labeling',fontsize=20)
	  plt.grid(True)

	  # plot hull
	  for h in range(len(hull_list)):
	    b_x = []
	    b_y = []
	    for hh in hull_list[h]:
	        b_x.append(tcoords[hh[1]])
	        b_y.append(flist[hh[0]])

	    mean_t = Features_stats[h][2]
	    mean_f = Features_stats[h][5]
	    etype = Features_stats[h][-1]

	    if etype == 'st':
	    	plt.plot(b_x,b_y,'g--')
	    else:
	    	plt.plot(b_x,b_y,'b--')
	    plt.plot(mds.epoch2num(mean_t),mean_f,'r*')   

	fig.autofmt_xdate()

	plt.savefig(curdir+'Spectrograms/'+str(first_file)+'_Features.png')
	plt.clf()
	plt.close()
	#plt.show()

	first_file = first_file+num_analysis_file

