# Imports
import numpy as np
from os import listdir

# Read in data
path = '/Volumes/icex6/ICEX_UNCLASS/ICEX16\
/macrura/2016-03-13/DURIP/DURIP_20160313T055853/'

directory = [f for f in listdir(path) if f.startswith("ACO")]

FS = 12000
NUM_SAMPLES = FS*2    
NUM_CHANNELS = 32

first_file = 2000+0*(1800)-1
last_file = first_file + (10)

aco_in = np.zeros((NUM_SAMPLES*(last_file-first_file), 32))

counter=0;
for i in np.arange(first_file,last_file):
 
    counter=counter+1;
    filename = path+directory[i];
    fid = open(filename, 'rb')

    data_temp = np.fromfile(filename, dtype='<f4',count=NUM_SAMPLES*NUM_CHANNELS)
    data_temp = np.reshape(data_temp,(NUM_CHANNELS,NUM_SAMPLES)).T

    #Read the single precision float acoustic data samples (in uPa)
    aco_in[((counter-1)*NUM_SAMPLES):(counter*NUM_SAMPLES),:] = data_temp
     
    fid.close()

time = (1/(FS))*np.arange(aco_in.shape[0])

