import numpy as np
from icex_load import icex_readin, training_set_form
import butter_bpfilt as bbpf
from sklearn import preprocessing
import matplotlib.pyplot as plt

# Read in data
path = '/Volumes/icex6/ICEX_UNCLASS/ICEX16\
/macrura/2016-03-13/DURIP/DURIP_20160313T055853/'
FS = 12000
NUM_CHN = 32
timesteps = 32
samples = 8192
overlap = 0.5

data, time = icex_readin(path,FS,NUM_CHN,2000,2150)
data_filt = bbpf.butter_bandpass_filter(data,40,1280,FS, order=6)
data_filt = preprocessing.scale(data_filt,axis=1)

train_dataset,t = training_set_form(data_filt,timesteps,NUM_CHN,samples,overlap)

plt.subplot(2,1,1)
plt.plot(time,data[:,16])
plt.xlabel('Time (sec)')
plt.ylabel('Data Amplitude')
plt.title('Raw Data')
plt.grid(True)
plt.axis('tight')

plt.subplot(2,1,2)
plt.plot(time, data_filt[:,16])
plt.xlabel('Time (sec)')
plt.ylabel('Data Amplitude')
plt.title('Filtered and Scaled Data')
plt.grid(True)
plt.axis('tight')
plt.show()

