############################### Imports ##################################################
import os
curdir = '/Users/Rui/Documents/Graduate/Research/ICEX:SIMI/lstm_eSelect/dev_new/'
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

from Feature import Feature, conjoin, proximity
import convex_hull
import copy

from beamform_3D import beamform_3D
from get_array_pitch import get_array_pitch
from save_obj import save_object

from scipy.signal import spectrogram,butter,freqz,lfilter


FS = 12000 # sampling frequency
NUM_CHANNELS = 32 # number of hydrophone channels
path = '/Volumes/icex6/ICEX_UNCLASS/ICEX16/macrura/2016-03-13/DURIP/\
DURIP_20160313T055853/'

n_fft = 4096
hop_length = int(n_fft/2)

first_file = 2274
last_file = first_file+60

noise_in,time_data = icex_readin(path,FS,NUM_CHANNELS,first_file,last_file) # read in stat and ana files centered around first_file

# b,a = butter(10, [100/6000,4096/6000], btype='band');
# w, h = freqz(b,a)
# # plt.plot((FS * 0.5 / np.pi) * w, abs(h))
# # plt.show()

# y = lfilter(b, a, noise_in[:,0])

num_nfft_noise = int(np.floor(np.shape(noise_in)[0]/hop_length))-1 # calculate number of total nfft bins for stats data
sxx = np.zeros((int(n_fft/2+1),num_nfft_noise,NUM_CHANNELS))

for c in range(NUM_CHANNELS):
	f,t,sxx[:,:,c] = spectrogram(noise_in[:,0], fs=FS, window=('hamming'), nperseg=n_fft, noverlap=hop_length, nfft=n_fft, detrend='constant', return_onesided=True, scaling='density', axis=-1, mode='psd')

sxx = np.mean(sxx,2)
# sxx = 10*np.log10(sxx)
sxxnew = sxx[0:1024,:]
#sxxnew= cv2.GaussianBlur(sxxnew,ksize=(5,5),sigmaX=1,sigmaY=1) 
plt.pcolormesh(t, f[0:1024], 10*np.log10(sxxnew),cmap='plasma')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.colorbar(format='%+2.0f')
plt.show()


xwin = 2
ywin = 90
thres = 5

sxxsmall = copy.deepcopy(sxxnew[xwin:-(xwin+1),ywin:-(ywin+1)])
sxxfilt = copy.deepcopy(sxxnew)
sxxfilt[0:xwin,:] = float('nan')
sxxfilt[-(xwin+1):-1,:] = float('nan')
sxxfilt[:,0:ywin] = float('nan')
sxxfilt[:,-(ywin+1):-1] = float('nan')

plt.pcolormesh(t, f[0:1024], 10*np.log10(sxxfilt),cmap='plasma')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.colorbar(format='%+2.0f')
plt.show()

for yind in np.arange(sxxsmall.shape[1])+ywin:
	for xind in np.arange(sxxsmall.shape[0])+xwin:

		# maxval = np.max(sxxnew[xind-xwin:xind+xwin,yind-ywin:yind+ywin])
		# val = np.abs(sxxfilt[xind,yind]-maxval)/maxval
		# if val>thres:
		# 	sxxfilt[xind,yind] = float('nan')
		# else:
		# 	sxxfilt[xind,yind] = val

		meanval = np.mean(sxxnew[xind-xwin:xind+xwin,yind-ywin:yind+ywin])
		val = sxxfilt[xind,yind]/meanval
		if val<thres:
			sxxfilt[xind,yind] = 0
		else:
			sxxfilt[xind,yind] = val

sxxfilt= cv2.GaussianBlur(sxxfilt,ksize=(25,25),sigmaX=1,sigmaY=1) 
plt.pcolormesh(t, f[0:1024], sxxfilt,cmap='plasma')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.colorbar(format='%+2.0f')
plt.show()



###################################### Get Noise Sample ##################################################
print('Calculating Background Noise ...')

first_file_noise = 2000
last_file_noise = 2900

noise_in,time_data = icex_readin(path,FS,NUM_CHANNELS,first_file_noise,last_file_noise) # read in stat and ana files centered around first_file
num_nfft_noise = int(np.floor(np.shape(noise_in)[0]/hop_length))-1 # calculate number of total nfft bins for stats data
S_noise = np.zeros((int(n_fft/2+1),num_nfft_noise,NUM_CHANNELS))

for c in range(NUM_CHANNELS):
  #print(c) # for each channel, determine stats data mel-spectrogram
	f,t,S_noise[:,:,c] = spectrogram(noise_in[:,0], fs=FS, window=('hamming'), nperseg=n_fft, noverlap=hop_length, nfft=n_fft, detrend='constant', return_onesided=True, scaling='density', axis=-1, mode='psd')
S_noise = np.mean(S_noise,axis=2) # Average over all channels
f_means = np.mean(S_noise,axis=1) # get mean of each f bin in stats data spectrogram
#f_vars = np.std(S_noise,axis=1) # get variance of each f bin in stats data spectrogram


####################################### Start Analysis #################################################

# Get data

print(first_file)
print('Getting data ... ')

aco_in,time_data = icex_readin(path,FS,NUM_CHANNELS,first_file+25,first_file+40) # read in stat and ana files centered around first_file
#time_data = dir_start_eptime+(first_file-1)*file_t_win+time_data # epoch time at start of imported data

# Plotting aco_in
#plt.plot(time_data,aco_in[:,15])
#plt.xlabel('Time (sec)')
#plt.ylabel('Data Amplitude')
#plt.show()

################################## Get Filtered Spectrogram #######################################
print('Filtering Spectrogram ... ')

# Noise Mel-spectrogram stats calculation

num_nfft_tot = int(np.floor(np.shape(aco_in)[0]/hop_length))-1 # calculate number of total nfft bins for stats data
S = np.zeros((int(n_fft/2+1),num_nfft_tot,NUM_CHANNELS))

for c in range(NUM_CHANNELS): # for each channel, determine stats data mel-spectrogram
	f,t,S[:,:,c] = spectrogram(aco_in[:,0], fs=FS, window=('hamming'), nperseg=n_fft, noverlap=hop_length, nfft=n_fft, detrend='constant', return_onesided=True, scaling='density', axis=-1, mode='psd')
S = np.mean(S,axis=2) # Average over all channels
#f_means = np.mean(S,axis=1) # get mean of each f bin in stats data spectrogram
#f_vars = np.std(S,axis=1) # get variance of each f bin in stats data spectrogram
# np.savetxt(curdir+'input/'+str(time_data[0])+'_input.txt',S)
# Analysis Data Mel-Spectrogram calculation

#num_nfft = int(np.ceil(np.shape(aco_in)[0]/hop_length)) # calculate number of total nfft bins for ana data
# flist = librosa.core.mel_frequencies(n_mels=n_mels, fmin=0.0, fmax=fmax, htk=False) # get list of frequencies of spectrogram y-axis
# tlist = (1/FS)*np.linspace(0,np.shape(aco_in)[0],num_nfft_tot) # get list of times for spectrogram x-axis
# tcoords = mds.epoch2num(tlist+time_data[0]) # get time coordinates for spectrogram x-axis (NOT EPOCH TIME!)

S = cv2.GaussianBlur(S,ksize=(7,7),sigmaX=1,sigmaY=1) 
S_ana_f = copy.deepcopy(S)

for row in range(S.shape[0]): # normalize each frequency bin to zero mean and unit variance
	S_ana_f[row,:] = (S[row,:]/f_means[row])#/f_vars[row] 

db_thres = 5
#S_ana_f[S_ana_f <= 0] = float('nan') # change all values below 0 to NaN
S_ana_log = copy.deepcopy(S_ana_f) # calculate log value of ana spectrogram
S_ana_log[S_ana_log <= db_thres] = float('nan') # set all values in S_ana_log that are < db_thres to NaN



plt.pcolormesh(t, f[0:1024], 10*np.log10(S[0:1024,:]),cmap='plasma')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.colorbar(format='%+2.0f')
plt.show()

from skimage.morphology import thin, skeletonize
S_skel = skeletonize(S_ana_f/np.max(S_ana_f))

plt.pcolormesh(t, f[0:1024], S_skel[0:1024,:],cmap='plasma')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
#plt.colorbar(format='%+2.0f')
plt.show()

