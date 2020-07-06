import numpy as np
from os import listdir

def sidex_readin(path, FS=1000, NUM_CHANNELS=16, first_file=0, last_file=1):
	directory = [f for f in np.sort(listdir(path)) if f.startswith("Sidex_20200125T")]
	NUM_SAMPLES = FS*60;
	aco_in = np.zeros((NUM_SAMPLES*(last_file-first_file), NUM_CHANNELS))

	counter=0;
	for i in np.arange(first_file,last_file):
		#print(counter)
		counter=counter+1
		filename = path+directory[i]
		#print(filename)
		data_temp = np.loadtxt(filename,delimiter=',',skiprows=2)
		aco_in[((counter-1)*NUM_SAMPLES):(counter*NUM_SAMPLES),:] = data_temp


	ffname = path+directory[first_file]	
	time = np.arange(0,(aco_in.shape[0]/FS),1/FS)

	return aco_in, time, ffname
