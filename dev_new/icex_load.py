import numpy as np
from os import listdir

def icex_readin(path, FS = 12000, NUM_CHANNELS = 32, first_file = 2000, last_file = 2450):
	directory = [f for f in np.sort(listdir(path)) if f.startswith("ACO")]
	NUM_SAMPLES = FS*2
	aco_in = np.zeros((NUM_SAMPLES*(last_file-first_file), NUM_CHANNELS))

	counter=0;
	for i in np.arange(first_file,last_file):
		counter=counter+1
		filename = path+directory[i]
		fid = open(filename, 'rb')

		data_temp = np.fromfile(filename, dtype='<f4',count=NUM_SAMPLES*NUM_CHANNELS)
		data_temp = np.reshape(data_temp,(NUM_CHANNELS,NUM_SAMPLES)).T
		#Read the single precision float acoustic data samples (in uPa)
		aco_in[((counter-1)*NUM_SAMPLES):(counter*NUM_SAMPLES),:] = data_temp

		fid.close()

	time = (1/(FS))*np.arange(aco_in.shape[0])

	return aco_in, time


def training_set_form(data_filt,timesteps = 32, chns = 32, samples = 8192, overlap = 0.5, FS = 12000):
	win_len = samples
	window_start = (timesteps+1)*overlap*np.round(win_len-win_len*overlap)
	step_start = np.round(win_len-win_len*overlap)
	num_window = np.floor(data_filt.shape[0]/window_start)-1
	t = []

	train_dataset = np.zeros((int(num_window),timesteps,chns,samples,1))

	for l in np.arange(int(num_window)):
		print(l+1, '/', int(num_window))
		t.append(((l+1)*window_start+1)/FS)
		data_seg = data_filt[int(l*window_start):int(l*window_start+(timesteps+1)*overlap*win_len),:]

		for i in np.arange(timesteps):
			train_dataset[l,i,:,:,0] = data_seg[int(i*step_start):int(i*step_start+win_len),:].T

	return train_dataset,t