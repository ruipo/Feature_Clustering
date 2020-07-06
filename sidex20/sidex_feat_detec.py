############################### Imports ##################################################
import os
curdir = '/Users/Rui/Documents/Graduate/Research/SIDEX/SIDEX20/feature_clustering/'
os.chdir(curdir)

import numpy as np
from sidex_load import sidex_readin
import matplotlib.pyplot as plt
import matplotlib.dates as mds
import librosa
import librosa.display
import cv2
import copy

path = '/Users/Rui/Documents/Graduate/Research/SIDEX/SIDEX20/sidex20_calibration/'
FS = 1000
NUM_CHANNELS = 16

n_fft = 512
hop_length = int(n_fft/2)
fmax = 128 # max frequency to examine
n_mels = 128 #

###################################### Get Noise Sample ##################################################
print('Calculating Background Noise ...')

first_file_noise = 0
last_file_noise = 30

noise_in,time_data = sidex_readin(path,FS,NUM_CHANNELS,first_file_noise,last_file_noise) # read in stat and ana files centered around first_file
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

###################################### Get data Sample ##################################################
first_file = 0
last_file = 10

data,time = sidex_readin(path,FS,NUM_CHANNELS,first_file,last_file)

t_ep_start = 1579980698.215000
time = t_ep_start+time
tcoords = mds.epoch2num(time)

tcalib = np.loadtxt(curdir+'calib_event_eptimes.txt')
tcalib_coords = mds.epoch2num(tcalib)


plt.plot(tcoords,data[:,0])
plt.plot(tcalib_coords,np.zeros(tcalib_coords.shape[0]),'r*')
ax = plt.gca()
ax.xaxis.set_major_formatter(formatter=mds.DateFormatter('%H:%M:%S'))
plt.grid()
plt.show()


num_nfft_tot = int(np.ceil(np.shape(data)[0]/hop_length)) # calculate number of total nfft bins for stats data
flist = librosa.core.mel_frequencies(n_mels=n_mels, fmin=0.0, fmax=fmax, htk=False)
tlist = (1/FS)*np.linspace(0,np.shape(data)[0],num_nfft_tot) # get list of times for spectrogram x-axis
tcoords = mds.epoch2num(tlist+t_ep_start)
S = np.zeros((n_mels,num_nfft_tot,NUM_CHANNELS)) # initialize stats data spectrogram matrix

for c in range(12): # for each channel, determine stats data mel-spectrogram
  S[:,:,c] = librosa.feature.melspectrogram(y=np.array(data[:,c]), sr=FS, n_fft=n_fft, hop_length=hop_length,fmax=fmax,n_mels=n_mels)
S = np.mean(S,axis=2)

S = cv2.GaussianBlur(S,ksize=(5,5),sigmaX=1,sigmaY=1) 
S_ana_f = copy.deepcopy(S)

for row in range(S.shape[0]): # normalize each frequency bin to zero mean and unit variance
	S_ana_f[row,:] = (S[row,:]/f_means[row])#/f_vars[row] 

S_ana_gradx = cv2.Sobel(S_ana_f,cv2.CV_64F,1,0,ksize=5)
S_ana_grady = cv2.Sobel(S_ana_f,cv2.CV_64F,0,1,ksize=5)
S_ana_gradl = np.maximum(np.abs(S_ana_gradx),np.abs(S_ana_grady))
S_ana_gradl = cv2.morphologyEx(S_ana_gradl, cv2.MORPH_OPEN, np.ones((3, 3)))
close_mask = cv2.morphologyEx(S_ana_gradl, cv2.MORPH_CLOSE, np.ones((5, 5)))

fig = plt.figure(figsize=(20,8))

ax1 = plt.subplot(2,4,1,autoscale_on=True)
librosa.display.specshow(10*np.log10(S),x_coords=tcoords, y_coords=flist,x_axis='time',y_axis='mel', sr=FS, fmax=fmax)
plt.plot(tcalib_coords,10*np.zeros(tcalib_coords.shape[0]),'r*')
cax = plt.colorbar()
cax.set_label('dB',labelpad=-30, y=1.05, rotation=0,fontsize=15)
ax = plt.gca()
ax.xaxis.set_major_formatter(formatter=mds.DateFormatter('%H:%M:%S'))
plt.xticks(fontsize=15)
plt.yticks(np.arange(0, fmax, 192),fontsize=15)
#plt.clim(110,160)
# plt.xlabel('Time')
# plt.ylabel('Frequency (Hz)')
# plt.title('Conventional')
ax1.set_xlabel('Time',fontsize=15)
ax1.set_ylabel('Frequency (Hz)',fontsize=15)
ax1.set_title('Conventional',fontsize=15)


ax2 = plt.subplot(2,4,2,sharey=ax1,autoscale_on=True)
librosa.display.specshow(S_ana_f,x_coords=tcoords, y_coords=flist,x_axis='time',y_axis='mel', sr=FS, fmax=fmax)
cax = plt.colorbar()
cax.set_label(' ',labelpad=-30, y=1.05, rotation=0)
ax = plt.gca()
ax.xaxis.set_major_formatter(formatter=mds.DateFormatter('%H:%M:%S'))
plt.xticks(fontsize=5)
plt.yticks(np.arange(0, fmax, 96),fontsize=5)
plt.clim(0,0.000005)
plt.xlabel('')
plt.ylabel('')
plt.title('F-Normalized')

ax3 = plt.subplot(2,4,3,sharey=ax1,autoscale_on=True)
librosa.display.specshow(np.abs(S_ana_gradx),x_coords=tcoords, y_coords=flist,x_axis='time',y_axis='mel', sr=FS, fmax=fmax)
cax = plt.colorbar()
cax.set_label(' ',labelpad=-30, y=1.05, rotation=0)
ax = plt.gca()
ax.xaxis.set_major_formatter(formatter=mds.DateFormatter('%H:%M:%S'))
plt.xticks(fontsize=5)
plt.yticks(np.arange(0, fmax, 96),fontsize=5)
#plt.clim(0,0.000005)
plt.xlabel('')
plt.ylabel('')
plt.title('xGradient')

ax4 = plt.subplot(2,4,4,sharey=ax1,autoscale_on=True)
librosa.display.specshow(np.abs(S_ana_grady),x_coords=tcoords, y_coords=flist,x_axis='time',y_axis='mel', sr=FS, fmax=fmax)
cax = plt.colorbar()
cax.set_label(' ',labelpad=-30, y=1.05, rotation=0)
ax = plt.gca()
ax.xaxis.set_major_formatter(formatter=mds.DateFormatter('%H:%M:%S'))
plt.xticks(fontsize=5)
plt.yticks(np.arange(0, fmax, 96),fontsize=5)
#plt.clim(0,50)
plt.xlabel('')
plt.ylabel('')
plt.title('yGradient')

ax5 = plt.subplot(2,4,5,sharey=ax1,autoscale_on=True)
librosa.display.specshow(S_ana_gradl,x_coords=tcoords, y_coords=flist,x_axis='time',y_axis='mel', sr=FS, fmax=fmax)
cax = plt.colorbar()
cax.set_label(' ',labelpad=-30, y=1.05, rotation=0)
ax = plt.gca()
ax.xaxis.set_major_formatter(formatter=mds.DateFormatter('%H:%M:%S'))
plt.xticks(fontsize=5)
plt.yticks(np.arange(0, fmax, 96),fontsize=5)
#plt.clim(0,10)
plt.xlabel('')
plt.ylabel('Frequency (Hz)')
plt.title('LGradient')

ax6 = plt.subplot(2,4,6,sharey=ax1,autoscale_on=True)
librosa.display.specshow(close_mask,x_coords=tcoords, y_coords=flist,x_axis='time',y_axis='mel', sr=FS, fmax=fmax)
cax = plt.colorbar()
cax.set_label(' ',labelpad=-30, y=1.05, rotation=0)
ax = plt.gca()
ax.xaxis.set_major_formatter(formatter=mds.DateFormatter('%H:%M:%S'))
plt.xticks(fontsize=5)
plt.yticks(np.arange(0, fmax, 96),fontsize=5)
plt.clim(0,1)
plt.xlabel('')
plt.ylabel('')
plt.title('mask')

plt.show()

# ax7 = plt.subplot(2,4,7,sharey=ax1,autoscale_on=True)
# librosa.display.specshow(S_ana_log,x_coords=tcoords, y_coords=flist,x_axis='time',y_axis='mel', sr=FS, fmax=fmax)
# cax = plt.colorbar()
# cax.set_label('dB',labelpad=-30, y=1.05, rotation=0,fontsize=15)
# ax = plt.gca()
# ax.xaxis.set_major_formatter(formatter=mds.DateFormatter('%H:%M:%S'))
# plt.xticks(fontsize=15)
# plt.yticks(np.arange(0, fmax, 192),fontsize=15)
# plt.clim(db_thres,2)
# # plt.xlabel('Time')
# # plt.ylabel('')
# # plt.title('Post-Processing')
# ax2.set_xlabel('Time',fontsize=15)
# ax2.set_ylabel('Frequency (Hz)',fontsize=15)

