############################### Imports ##################################################
import os
curdir = '/Users/Rui/Documents/Graduate/Research/Feature_clustering/dev_new/'
os.chdir(curdir)

import numpy as np
import time
import calendar

from icex_load import icex_readin
import matplotlib.pyplot as plt
import matplotlib.dates as mds

import librosa
import librosa.display
import cv2

from Feature import conjoin
from feature_detection import feature_detection
from h_clustering import h_clustering
import convex_hull
import copy

from beamform_3D import beamform_3D
from get_array_pitch import get_array_pitch
from save_obj import save_object

#exec(open('get_imports.py').read())

##################################### Get Array Pitch ##########################################
print('Determining Array Pitch w. time ... ')
ep2pitch = get_array_pitch(curdir+'array_pitch/')

####################################### Set Analysis Constants ###################################
file_t_win = 2 # FOR ACO DATA, each file is 2 seconds of data
#noise_stat_t_win = 30 # noise stat calc window in seconds
analysis_t_win = 30 # analysis window length in seconds
FS = 12000 # sampling frequency
NUM_CHANNELS = 32 # number of hydrophone channels
dataday = '2016-03-13' # the date of the current data
# Data Folder PATH
path = '/Volumes/icex6/ICEX_UNCLASS/ICEX16/macrura/2016-03-13/DURIP/\
DURIP_20160313T055853/'
# epoch time at start of the data directory
dir_start_eptime = calendar.timegm(time.strptime(dataday+' 05:58:53',\
'%Y-%m-%d %H:%M:%S'))

# Define first and last files to analyze in the directory
first_file = 1999
last_file = 2100

#num_stats_file = int(np.ceil(noise_stat_t_win/file_t_win)) # number of files to use for stats
num_analysis_file = int(np.ceil(analysis_t_win/file_t_win)) # number of files to use for analysis

# Spectrogram Variables
n_fft = 4096
hop_length = int(n_fft/2)
fmax = 2048 # max frequency to examine
n_mels = 128 # number of frequency bin in mel-spectrogram

# Feature Detection Variables
area_thres = 5 # min size of Feature Area to save a Feature
prox_thres = 5.5 # starting proximity distance when grouping Features together
db_thres = 1.5 # value above db_thres number of times f_mean

# Beamform Feature
# Array geometry
plocs = np.array([[0,0,15.3750],[0,0,13.8750],[0,0,12.3750],[0,0,10.8750],[0,0,9.3750],[0,0,7.8750],[0,0,7.1250],[0,0,6.3750],[0,0,5.6250],[0,0,4.8750],[0,0,4.1250],[0,0,3.3750],[0,0,2.6250],[0,0,1.8750],[0,0,1.1250],[0,0,0.3750],[0,0,-0.3750],[0,0,-1.1250],[0,0,-1.8750],[0,0,-2.6250],[0,0,-3.3750],[0,0,-4.1250],[0,0,-4.8750],[0,0,-5.6250],[0,0,-6.3750],[0,0,-7.1250],[0,0,-7.8750],[0,0,-9.3750],[0,0,-10.8750],[0,0,-12.3750],[0,0,-13.8750],[0,0,-15.3750]])
elev = np.arange(-90,91,1) 
az = 0
propgaspeed = 1435 # sound speed
overlap = 0.5
weighting= 'icex_hanning'

###################################### Get Noise Sample ##################################################
print('Calculating Background Noise ...')

first_file_noise = 2000
last_file_noise = 2900

noise_in,time_data = icex_readin(path,FS,NUM_CHANNELS,first_file_noise,last_file_noise) # read in stat and ana files centered around first_file
num_nfft_noise = int(np.ceil(np.shape(noise_in)[0]/hop_length)) # calculate number of total nfft bins for stats data
S_noise = np.zeros((n_mels,num_nfft_noise,NUM_CHANNELS)) # initialize stats data spectrogram matrix

for c in range(NUM_CHANNELS):
  #print(c) # for each channel, determine stats data mel-spectrogram
  S_noise[:,:,c] = librosa.feature.melspectrogram(y=np.array(noise_in[:,c]), sr=FS, n_fft=n_fft, hop_length=hop_length,fmax=fmax,n_mels=n_mels)

S_noise_start = np.mean(S_noise,axis=2) # Average over all channels
f_means_start = np.mean(S_noise,axis=1) # get mean of each f bin in stats data spectrogram
#f_vars = np.std(S_noise,axis=1) # get variance of each f bin in stats data spectrogram

S_noise = np.mean(S_noise,axis=2) # Average over all channels
f_means = np.mean(S_noise,axis=1)

####################################### Start Analysis #################################################
# Get data
while first_file < last_file:
	print('Working on file: ', first_file)
	print('Getting data ... ')

	aco_in,time_data = icex_readin(path,FS,NUM_CHANNELS,first_file,first_file+num_analysis_file) # read in stat and ana files centered around first_file
	time_data = dir_start_eptime+(first_file-1)*file_t_win+time_data # epoch time at start of imported data

	# Plotting aco_in
	#plt.plot(time_data,aco_in[:,15])
	#plt.xlabel('Time (sec)')
	#plt.ylabel('Data Amplitude')
	#plt.show()

	################################## Get Filtered Spectrogram #######################################
	print('Filtering Spectrogram ... ')

	# Noise Mel-spectrogram stats calculation

	num_nfft_tot = int(np.ceil(np.shape(aco_in)[0]/hop_length)) # calculate number of total nfft bins for stats data
	S = np.zeros((n_mels,num_nfft_tot,NUM_CHANNELS)) # initialize stats data spectrogram matrix

	for c in range(NUM_CHANNELS): # for each channel, determine stats data mel-spectrogram
	  S[:,:,c] = librosa.feature.melspectrogram(y=np.array(aco_in[:,c]), sr=FS, n_fft=n_fft, hop_length=hop_length,fmax=fmax,n_mels=n_mels)
	S = np.mean(S,axis=2) # Average over all channels
	#f_means = np.mean(S,axis=1) # get mean of each f bin in stats data spectrogram
	#f_vars = np.std(S,axis=1) # get variance of each f bin in stats data spectrogram
	np.savetxt(curdir+'input/'+str(time_data[0])+'_input.txt',S)

	# Analysis Data Mel-Spectrogram calculation

	#num_nfft = int(np.ceil(np.shape(aco_in)[0]/hop_length)) # calculate number of total nfft bins for ana data
	flist = librosa.core.mel_frequencies(n_mels=n_mels, fmin=0.0, fmax=fmax, htk=False) # get list of frequencies of spectrogram y-axis
	tlist = (1/FS)*np.linspace(0,np.shape(aco_in)[0],num_nfft_tot) # get list of times for spectrogram x-axis
	tcoords = mds.epoch2num(tlist+time_data[0]) # get time coordinates for spectrogram x-axis (NOT EPOCH TIME!)

	S = cv2.GaussianBlur(S,ksize=(5,5),sigmaX=1,sigmaY=1) 
	S_ana_f = copy.deepcopy(S)

	for row in range(S.shape[0]): # normalize each frequency bin to zero mean and unit variance
		S_ana_f[row,:] = (S[row,:]/f_means[row])#/f_vars[row] 

	S_ana_gradx = cv2.Sobel(S_ana_f,cv2.CV_64F,1,0,ksize=5)
	S_ana_grady = cv2.Sobel(S_ana_f,cv2.CV_64F,0,1,ksize=5)
	S_ana_gradl = np.maximum(np.abs(S_ana_gradx),np.abs(S_ana_grady))
	S_ana_gradl = cv2.morphologyEx(S_ana_gradl, cv2.MORPH_OPEN, np.ones((3, 3)))
	close_mask = cv2.morphologyEx(S_ana_gradl, cv2.MORPH_CLOSE, np.ones((5, 5)))

	close_mask[close_mask<=8] = 0
	close_mask[close_mask!=0] = 1

	#S_ana_f[S_ana_f <= 0] = float('nan') # change all values below 0 to NaN
	S_ana_log = copy.deepcopy(S_ana_f) # calculate log value of ana spectrogram
	S_ana_log = S_ana_log*close_mask
	#S_ana_logt = copy.deepcopy(S_ana_log)
	S_ana_log[S_ana_log <= db_thres] = float('nan') # set all values in S_ana_log that are < db_thres to NaN
	S_ana_log[0:9,:] = float('nan')

	# S_ana_logt[S_ana_logt <= db_thres] = 0
	# S_ana_logt[S_ana_logt != 0] = 1
	# S_ana_logt[0:9,:] = 0

	# from skimage.morphology import skeletonize
	# S_skel = skeletonize(S_ana_logt)

	# S_sino = copy.deepcopy(S_ana_logt)
	# S_sino[S_skel==False] = 0

	# from skimage.transform import radon
	# theta = np.linspace(0., 180., 180, endpoint=False)

	# S_sino1 = S_sino[0:128,60:75]
	# sinogram = radon(S_sino1,theta=theta,circle=True)

	# fig = plt.figure(figsize=(20,8))
	# ax1 = plt.subplot(1,2,1,autoscale_on=True)
	# librosa.display.specshow(S_sino1,x_coords=tcoords[60:75], y_coords=flist[0:128],x_axis='time',y_axis='mel', sr=FS, fmax=fmax)
	# ax1.plot(S_sino1.shape[0]/2,S_sino1.shape[1]/2,'*')
	# ax = plt.gca()
	# ax.xaxis.set_major_formatter(formatter=mds.DateFormatter('%H:%M:%S'))
	# #plt.yticks(np.arange(0, fmax, 96))
	# plt.clim(0,1)
	# plt.xlabel('')
	# plt.ylabel('')
	# plt.title('Skeleton')
	# ax2 = plt.subplot(1,2,2,autoscale_on=True)
	# ax2.set_title("Radon transform\n(Sinogram)")
	# ax2.set_xlabel("Projection angle (deg)")
	# ax2.set_ylabel("Projection position (pixels)")
	# ax2.pcolormesh(np.arange(0,180),np.flipud(np.arange(-sinogram.shape[0]/2,sinogram.shape[0]/2)),sinogram, cmap=plt.cm.Greys_r)
	# plt.show()

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
	  if ent.area <= area_thres: # if Feature area <= area_thres, set Feature pixels to NaN
	    for p in ent.pixels:
	    	S_g_log_test[p] = np.float('nan')
	  else: # else, keep the Feature, add to fFlist, determine its stats, calculate the hull of the Feature
	    for p in ent.pixels:
	      S_noise_cur[p] = np.float('nan')

	    filtered_Features_list.append(ent)
	    Features_stats.append(ent.stats(flist,tcoords))
	    hull = convex_hull.convex_hull(np.array(ent.pixels).T)
	    hull = np.vstack((hull, hull[0]))
	    hull_list.append(hull)

	# Set noise level for next iteration
	f_means_cur = np.nanmean(S_noise_cur,axis=1)
	#f_vars_cur = np.nanvar(S_noise_cur,axis=1)
	f_means = 0.5*f_means+0.5*f_means_cur
	#f_vars = 0.5*f_vars+0.5*f_vars_cur

	#np.savetxt(curdir+'output/'+str(time_data[0])+'_output.txt',~np.isnan(S_g_log_test))
	# Plot Saved Features
	fig = plt.figure(figsize=(20,8))

	ax1 = plt.subplot(1,3,1,autoscale_on=True)
	librosa.display.specshow(10*np.log10(S),x_coords=tcoords, y_coords=flist,x_axis='time',y_axis='mel', sr=FS, fmax=fmax)
	cax = plt.colorbar()
	cax.set_label('dB',labelpad=-30, y=1.05, rotation=0,fontsize=15)
	ax = plt.gca()
	ax.xaxis.set_major_formatter(formatter=mds.DateFormatter('%H:%M:%S'))
	plt.xticks(fontsize=15)
	plt.yticks(np.arange(0, fmax, 192),fontsize=15)
	plt.clim(110,160)
	# plt.xlabel('Time')
	# plt.ylabel('Frequency (Hz)')
	# plt.title('Conventional')
	ax1.set_xlabel('Time',fontsize=15)
	ax1.set_ylabel('Frequency (Hz)',fontsize=15)
	ax1.set_title('Conventional',fontsize=15)

	# ax2 = plt.subplot(2,4,2,sharey=ax1,autoscale_on=True)
	# librosa.display.specshow(S_ana_f,x_coords=tcoords, y_coords=flist,x_axis='time',y_axis='mel', sr=FS, fmax=fmax)
	# cax = plt.colorbar()
	# cax.set_label(' ',labelpad=-30, y=1.05, rotation=0)
	# ax = plt.gca()
	# ax.xaxis.set_major_formatter(formatter=mds.DateFormatter('%H:%M:%S'))
	# plt.xticks(fontsize=5)
	# plt.yticks(np.arange(0, fmax, 96),fontsize=5)
	# plt.clim(0,5)
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
	# plt.yticks(np.arange(0, fmax, 96),fontsize=5)
	# plt.clim(0,50)
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
	# plt.yticks(np.arange(0, fmax, 96),fontsize=5)
	# plt.clim(0,50)
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
	# plt.yticks(np.arange(0, fmax, 96),fontsize=5)
	# plt.clim(0,10)
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
	# plt.yticks(np.arange(0, fmax, 96),fontsize=5)
	# plt.clim(0,1)
	# plt.xlabel('')
	# plt.ylabel('')
	# plt.title('mask')

	ax2 = plt.subplot(1,3,2,sharey=ax1,autoscale_on=True)
	librosa.display.specshow(S_ana_log,x_coords=tcoords, y_coords=flist,x_axis='time',y_axis='mel', sr=FS, fmax=fmax)
	cax = plt.colorbar()
	cax.set_label('dB',labelpad=-30, y=1.05, rotation=0,fontsize=15)
	ax = plt.gca()
	ax.xaxis.set_major_formatter(formatter=mds.DateFormatter('%H:%M:%S'))
	plt.xticks(fontsize=15)
	plt.yticks(np.arange(0, fmax, 192),fontsize=15)
	plt.clim(db_thres,2)
	# plt.xlabel('Time')
	# plt.ylabel('')
	# plt.title('Post-Processing')
	ax2.set_xlabel('Time',fontsize=15)
	ax2.set_ylabel('Frequency (Hz)',fontsize=15)
	ax2.set_title('Post-Processing',fontsize=15)

	# ax8 = plt.subplot(2,4,8,sharey=ax1,autoscale_on=True)
	# librosa.display.specshow(S_sino,x_coords=tcoords, y_coords=flist,x_axis='time',y_axis='mel', sr=FS, fmax=fmax)
	# cax = plt.colorbar()
	# cax.set_label(' ',labelpad=-30, y=1.05, rotation=0)
	# ax = plt.gca()
	# ax.xaxis.set_major_formatter(formatter=mds.DateFormatter('%H:%M:%S'))
	# plt.xticks(fontsize=5)
	# plt.yticks(np.arange(0, fmax, 96),fontsize=5)
	# plt.clim(0,1)
	# plt.xlabel('')
	# plt.ylabel('')
	# plt.title('Skeleton')

	ax3 = plt.subplot(1,3,3,sharey=ax1,autoscale_on=True)
	if np.where(~np.isnan(S_g_log_test))[0].shape[0]>0:
	     
	  librosa.display.specshow(S_g_log_test,x_coords=tcoords, y_coords=flist,x_axis='time', y_axis='mel', sr=FS, fmax=fmax)
	  cax = plt.colorbar()
	  cax.set_label('dB',labelpad=-30, y=1.05, rotation=0,fontsize=15)
	  ax = plt.gca()
	  ax.xaxis.set_major_formatter(formatter=mds.DateFormatter('%H:%M:%S'))
	  plt.xticks(fontsize=15)
	  plt.yticks(np.arange(0, fmax, 192),fontsize=15)
	  plt.clim(db_thres,2)
	  #plt.xlabel('Time')
	  #plt.ylabel('')
	  #plt.title('Post-Clustering')
	  ax3.set_xlabel('Time',fontsize=15)
	  ax3.set_ylabel('Frequency (Hz)',fontsize=15)
	  ax3.set_title('Post-Clustering',fontsize=15)

	  # plot hull
	  for h in range(len(hull_list)):
	    b_x = []
	    b_y = []
	    for hh in hull_list[h]:
	        b_x.append(tcoords[hh[1]])
	        b_y.append(flist[hh[0]])

	    mean_t = Features_stats[h][2]
	    mean_f = Features_stats[h][5]

	    plt.plot(b_x,b_y,'g--')
	    plt.plot(mds.epoch2num(mean_t),mean_f,'r*')   

	fig.autofmt_xdate()
	plt.savefig(curdir+'Spectrograms_prox/'+str(first_file)+'_Features.png') # Save saved Feature figure
	plt.clf()
	plt.close()

	########################################## Beamform each Saved Feature ##############################################

	# Features_stats (# of features by 6 (maxt,mint,meant,maxf,minf,meanf))
	print('Beamforming Features ... ')
	for ff in range(len(filtered_Features_list)): # for each Feature stat in Features_stats matrix
		try:

		  Feat_current = filtered_Features_list[ff] # Define current Feature
		  Feat = Features_stats[ff] # Define Feature stats
		  start_ind = np.argmin(np.abs(Feat[1]-time_data)) # find start ind
		  end_ind = np.argmin(np.abs(Feat[0]-time_data)) # find end ind

		  # get pitch at time of Feature
		  t_pitch = (Feat[0]+Feat[1])/2 
		  pitch_ind = np.argmin(np.abs(t_pitch-time_data))
		  pitch = ep2pitch(t_pitch)

		  f_range = (Feat[4],Feat[3]) # define f_range for beamforming

		  # define hydrophone locations for pitched array
		  p_pitch = copy.deepcopy(plocs)
		  p_pitch[:,0] = plocs[:,2]*np.cos(np.deg2rad(90-pitch))
		  p_pitch[:,2] = plocs[:,2]*np.sin(np.deg2rad(90-pitch))

		  # Get Feature data
		  feat_data = aco_in[start_ind:end_ind,:]

		  if np.any(feat_data):
		    # Define Beamform FFT NFFT
		    if np.shape(feat_data)[0] < 512:
		      NFFT = np.shape(feat_data)[0]
		    else:
		      NFFT = 512

		    # Define Beamform FFT window
		    fft_window = np.hanning(n_fft+2)
		    fft_window = np.delete(fft_window,[0,n_fft+1])

		    # Beamform
		    beamform_output,tvec,fvec = beamform_3D(feat_data, p_pitch, FS, elev, az, propgaspeed, f_range, fft_window, NFFT, overlap=overlap, weighting=weighting)
		    beamform_output_avgt = np.nanmean(np.nanmean(10*np.log10(beamform_output),axis=2),axis=0) # average over az, time
		    beamform_output_avg = np.mean(beamform_output_avgt,axis=1) # average of az, time, frequency
		    
		    # find max elevation angle
		    indmax = np.argmax(beamform_output_avg) 
		    Feat_current.elevmax = elev[indmax]
		    save_object(Feat_current,curdir+'Features/'+str(Feat_current.start_t)+'.pkl')

		    # Plot
		    fig1 = plt.figure(figsize=(20,8))

		    plt.subplot(1,2,1,autoscale_on=True)
		    plt.pcolormesh(fvec,elev,beamform_output_avgt)
		    cax = plt.colorbar()
		    cax.set_label('dB',labelpad=-30, y=1.05, rotation=0)
		    plt.clim(30,70)
		    plt.ylim(-90,90)
		    plt.xlabel('Frequency (Hz)')
		    plt.ylabel('Elevation (Deg.)')
		    plt.title(str(Feat_current.start_t))

		    plt.subplot(1,2,2,autoscale_on=True)
		    plt.plot(beamform_output_avg,elev)
		    plt.xlabel('Power (dB)')
		    plt.ylabel('')
		    plt.ylim(-90,90)
		    plt.title(str(Feat_current.start_t))
		    plt.grid()

		    plt.savefig(curdir+'Beamforming/'+str(Feat_current.start_t)+'_bf.png')
		    plt.clf()
		    plt.close()
		  
		  else:
		    continue

		except:
			pass

	first_file = first_file+num_analysis_file

