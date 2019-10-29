# Imports
import numpy as np
from os import listdir
import matplotlib.pyplot as plt 
from scipy.signal import butter, filtfilt, freqz

# Read in data
path = '/Volumes/icex6/ICEX_UNCLASS/ICEX16\
/macrura/2016-03-13/DURIP/DURIP_20160313T055853/'

directory = [f for f in listdir(path) if f.startswith("ACO")]

FS = 12000
NUM_SAMPLES = FS*2    
NUM_CHANNELS = 32

first_file = 2000+0*(1800)-1
last_file = first_file + (450)

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

plt.plot(time,aco_in[:,16])
plt.xlabel('Time (sec)')
plt.ylabel('Data Amplitude')
plt.show()

# Filter Data
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    # w, h = freqz(b, a, worN=2000)
    # plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Gain')
    # plt.grid(True)
    # plt.legend(loc='best')
    # plt.show()
    return y

data_filt = np.zeros(np.shape(aco_in))   
for c in range(NUM_CHANNELS):
	data_filt[:,c] = butter_bandpass_filter(aco_in[:,c], 40, 1280, FS, order=6)

from sklearn import preprocessing
data_filt = preprocessing.scale(data_filt,axis=1)

plt.plot(time, data_filt[:,16])
plt.xlabel('Time (sec)')
plt.grid(True)
plt.axis('tight')
plt.show()

# Format data into training set
timesteps = 32
chns = NUM_CHANNELS
samples = 8192
overlap = 0.5

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

