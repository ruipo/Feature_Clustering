# Read in icex data from user input. Plots the data files selected by the user. 

import numpy as np
from os import listdir,chdir

curdir = '/Volumes/Backup Plus/DURIP_bench_test/DURIP_20200310T200051/'
chdir(curdir)

from scipy.signal import spectrogram
import matplotlib.pyplot as plt 
from beamform_3D import beamform_3D

while True:
	try:
		path = input('Enter the path to data directory. Ex: /Users/Rui/Desktop/ : ')
		directory = [f for f in listdir(path) if f.startswith("ACO")]
	except:
		print("Invalid directory input!")
		continue
	if len(directory) < 1:
		print("No data files in directory!")
		continue
	else:
		break

while True:
	try:
		FS = int(input('Enter sampling frequency in Hz. For default, enter 12000. : '))
	except:
		print("Invalid input!")
		continue
	else:
		break

while True:
	try:		
		NUM_CHANNELS = int(input('Enter number of channels in array. For default, enter 32. : '))
	except:
		print("Invalid input!")
		continue
	else:
		break

while True:
	try:	
		first_file = int(input('Enter the first file to be plotted. For default, enter 1. : '))
	except:
		print("Invalid input!")
		continue
	else:
		break

while True:
	try:
		last_file = int(input('Enter the last file to be plotted. For default (2min of data), add 61 to first file. For all files, enter 0. : '))
	except:
		print("Invalid input!")
		continue
	if last_file <= first_file and last_file > 0:
		print("last_file must be greater than first_file!")
		continue
	else:
		break

if len(directory) < last_file:
	last_file = len(directory)
	print("There are not that many files in directory; showing all data in directory.")

if last_file == 0:
	first_file = 1
	last_file = len(directory)
	print("Showing all data in directory")

while True:
	try:
		show_chn = int(input('Enter the channel to be plotted. For all, enter 0 : '))
	except:
		print("Invalid input!")
		continue
	if show_chn > NUM_CHANNELS:
		print("Channel number must be less than # of channels!")
		continue
	else:
		break

while True:
	try:
		f_range_low = int(input('Enter lower frequency limit : '))
	except:
		print("Invalid input!")
		continue
	else:
		break

while True:
	try:
		f_range_high = int(input('Enter upper frequency limit. max 6000 : '))
	except:
		print("Invalid input!")
		continue
	if f_range_high < f_range_low:
		print("Upper frequency limit must be greater than lower frequency limit!")
		continue
	else:
		break

while True:
	try:
		c_range_low = int(input('Enter lower db limit : '))
	except:
		print("Invalid input!")
		continue
	else:
		break

while True:
	try:
		c_range_high = int(input('Enter upper db limit : '))
	except:
		print("Invalid input!")
		continue
	if c_range_high < c_range_low:
		print("Upper db limit must be greater than lower db limit!")
		continue
	else:
		break

print('PROCESSING...')

NUM_SAMPLES = FS*2    

aco_in = np.zeros((NUM_SAMPLES*(last_file-first_file), NUM_CHANNELS))

counter=0;
for i in np.arange(first_file,last_file):
 
    counter=counter+1;
    filename = path+directory[i];
    fid = open(filename, 'rb')

    data_temp = np.fromfile(filename, dtype='ieee-le',count=NUM_SAMPLES*NUM_CHANNELS)
    data_temp = np.reshape(data_temp,(NUM_CHANNELS,NUM_SAMPLES)).T

    #Read the single precision float acoustic data samples (in uPa)
    aco_in[((counter-1)*NUM_SAMPLES):(counter*NUM_SAMPLES),:] = data_temp
     
    fid.close()

time = (1/(FS))*np.arange(aco_in.shape[0])

plocs = np.array([[0,0,15.3750],[0,0,13.8750],[0,0,12.3750],[0,0,10.8750],[0,0,9.3750],[0,0,7.8750],[0,0,7.1250],[0,0,6.3750],[0,0,5.6250],[0,0,4.8750],[0,0,4.1250],[0,0,3.3750],[0,0,2.6250],[0,0,1.8750],[0,0,1.1250],[0,0,0.3750],[0,0,-0.3750],[0,0,-1.1250],[0,0,-1.8750],[0,0,-2.6250],[0,0,-3.3750],[0,0,-4.1250],[0,0,-4.8750],[0,0,-5.6250],[0,0,-6.3750],[0,0,-7.1250],[0,0,-7.8750],[0,0,-9.3750],[0,0,-10.8750],[0,0,-12.3750],[0,0,-13.8750],[0,0,-15.3750]])
elev = np.arange(-90,90,1) 
az = 0
propgaspeed = 1435 # sound speed
overlap = 0.5
weighting = 'icex_hanning'
f_range = (f_range_low,f_range_high)
fft_window = np.hanning(8194)
fft_window = np.delete(fft_window,[0,8193])
NFFT = 1024

beamform_output,tvec,fvec = beamform_3D(aco_in, plocs, FS, elev, az, propgaspeed, f_range, fft_window, NFFT, overlap=overlap, weighting=weighting)
beamform_output_avgt = np.nanmean(np.nanmean(10*np.log10(beamform_output),axis=2),axis=0) # average over az, time
beamform_output_avg = np.mean(beamform_output_avgt,axis=1) # average of az, time, frequency
      
if show_chn == 0:
	for chn in np.arange(NUM_CHANNELS):
		f, t, Sxx = spectrogram(aco_in[:,chn], FS, nperseg=4096, noverlap=2048, nfft=4096)

		print('PLOTTING...')
		plt.figure(figsize=(20,8))

		plt.subplot(2,2,1, autoscale_on=True)
		plt.plot(time,aco_in[:,chn])
		plt.xlim(0,time[-1])
		plt.xlabel('Time (sec)')
		plt.ylabel('Data Amplitude')
		plt.title('Data on chn %d' %(chn+1))
		plt.grid()

		plt.subplot(2,2,2, autoscale_on=True)
		plt.pcolormesh(t, f, 10*np.log10(Sxx),vmin=c_range_low,vmax=c_range_high)
		plt.ylabel('Frequency [Hz]')
		plt.xlabel('Time (sec)')
		plt.ylim(f_range_low,f_range_high)
		cax = plt.colorbar()
		cax.set_label('dB')

		plt.subplot(2,2,4,autoscale_on=True)
		plt.pcolormesh(fvec,elev,beamform_output_avgt,vmin=c_range_low,vmax=c_range_high)
		cax = plt.colorbar()
		cax.set_label('dB')
		plt.ylim(-90,90)
		plt.xlabel('Frequency (Hz)')
		plt.ylabel('Elevation (Deg.)')

		plt.subplot(2,2,3,autoscale_on=True)
		plt.plot(beamform_output_avg,elev)
		plt.xlabel('Power (dB)')
		plt.ylabel('')
		plt.ylim(-90,90)
		plt.grid()

		plt.show()

else:
	f, t, Sxx = spectrogram(aco_in[:,show_chn-1], FS, nperseg=4096, noverlap=2048, nfft=4096)
	print('PLOTTING...')
	plt.figure(figsize=(20,8))

	plt.subplot(2,2,1, autoscale_on=True)
	plt.plot(time,aco_in[:,show_chn-1])
	plt.xlim(0,time[-1])
	plt.xlabel('Time (sec)')
	plt.ylabel('Data Amplitude')
	plt.title('Data on chn %d' %(show_chn))
	plt.grid()

	plt.subplot(2,2,2, autoscale_on=True)
	plt.pcolormesh(t, f, 10*np.log10(Sxx),vmin=c_range_low,vmax=c_range_high)
	plt.ylabel('Frequency [Hz]')
	plt.xlabel('Time (sec)')
	plt.ylim(f_range_low,f_range_high)
	cax = plt.colorbar()
	cax.set_label('dB')

	plt.subplot(2,2,4,autoscale_on=True)
	plt.pcolormesh(fvec,elev,beamform_output_avgt,vmin=c_range_low,vmax=c_range_high)
	cax = plt.colorbar()
	cax.set_label('dB')
	plt.ylim(-90,90)
	plt.xlabel('Frequency (Hz)')
	plt.ylabel('Elevation (Deg.)')

	plt.subplot(2,2,3,autoscale_on=True)
	plt.plot(beamform_output_avg,elev)
	plt.xlabel('Power (dB)')
	plt.ylabel('')
	plt.ylim(-90,90)
	plt.grid()

	plt.show()





