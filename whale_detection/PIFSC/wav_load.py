import numpy as np 
from os import listdir
from librosa import load

def wav_readin(path, first_file=0, last_file=1):
	directory = [f for f in np.sort(listdir(path)) if f.endswith(".wav")]
	aco_in = np.array([]);

	for i in np.arange(first_file,last_file):
		filename = path+directory[i]
		data_temp,FS = load(filename,sr=None,mono=False)

		aco_in = np.hstack([aco_in, data_temp]) if aco_in.size else data_temp


	ffname = path+directory[first_file]
	time = np.arange(0,(np.shape(aco_in)[1]/FS),1/FS)
	NUM_CHANNELS = aco_in.shape[0]

	return aco_in, time, FS, NUM_CHANNELS, ffname