# Imports
import numpy as np
from os import listdir
import matplotlib.pyplot as plt 
from scipy.signal import butter, filtfilt, spectrogram, convolve2d
import librosa
import librosa.display
import cv2

# Read in data
path = '/Volumes/icex6/ICEX_UNCLASS/ICEX16\
/macrura/2016-03-13/DURIP/DURIP_20160313T055853/'

directory = [f for f in listdir(path) if f.startswith("ACO")]

FS = 12000
NUM_SAMPLES = FS*2    
NUM_CHANNELS = 32

first_file = 2000+0*(1800)-1
last_file = first_file + (150)

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

# Filter Data and plot spectrogram
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
	data_filt[:,c] = butter_bandpass_filter(aco_in[:,c], 320, 640, FS, order=6)

from sklearn import preprocessing
data_filt = preprocessing.scale(data_filt,axis=1)

plt.plot(time,data_filt)
plt.xlabel('Time (sec)')
plt.grid(True)
plt.axis('tight')
plt.show()

f, t, Sxx = spectrogram(aco_in[:,16], FS, window='hamming',nperseg=8192, noverlap=4096, nfft=8192)
plt.pcolormesh(t, f, 10*np.log10(Sxx), vmin=50, vmax=80)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.colorbar()
plt.show()


# Plot filtered and smoothed mel_spectrograms and gradients
scharr = np.array([[ -3-3j, 0-10j,  +3 -3j],[-10+0j, 0+ 0j, +10 +0j],[ -3+3j, 0+10j,  +3 +3j]]) # Gx + j*Gy
for i in np.arange(0,10,0.5):
    S = np.zeros((128,128,NUM_CHANNELS))
    for c in range(NUM_CHANNELS):
        y = np.array(aco_in[int(i*(8*8192)):int((i+1)*(8*8192)),c])
        S[:,:,c] = librosa.feature.melspectrogram(y=y, sr=FS, n_fft=1024, hop_length=512,fmax=4096)
        for row in range(S.shape[0]):
            S[row,:,c] = S[row,:,c]/np.max(S[row,:,c])

    S = np.prod(S,axis=2)**(1/NUM_CHANNELS)*255
    S_grad = np.absolute(convolve2d(S, scharr, boundary='symm', mode='same'))
    S_grad = S_grad/np.max(np.max(S_grad))*255

    S = np.float32(S)
    S_grad = np.float32(S_grad)

    S_filt = cv2.bilateralFilter(S,15,25,25)
    S_grad = cv2.bilateralFilter(S_grad,15,25,25)

    librosa.display.specshow(S_filt, x_axis='time',y_axis='mel', sr=FS,fmax=4096)
    plt.colorbar()
    plt.clim(0,255)
    plt.title('Mel-frequency spectrogram')
    plt.tight_layout()
    plt.savefig('/Users/Rui/Documents/Graduate/Research/ICEX:SIMI/lstm_eSelect/results/figure_'+str(i)+'.png')
    plt.clf()

    librosa.display.specshow(S_grad, x_axis='time',y_axis='mel', sr=FS,fmax=4096)
    plt.colorbar()
    plt.clim(0,255)
    plt.title('Mel-frequency spectrogram')
    plt.tight_layout()
    plt.savefig('/Users/Rui/Documents/Graduate/Research/ICEX:SIMI/lstm_eSelect/results1/figure_'+str(i)+'.png')
    plt.clf()


# Beamform first, then plot mel-spectrogram
p = np.array([[0,0,15.3750],[0,0,13.8750],[0,0,12.3750],[0,0,10.8750],[0,0,9.3750],[0,0,7.8750],[0,0,7.1250],[0,0,6.3750],[0,0,5.6250],[0,0,4.8750],[0,0,4.1250],[0,0,3.3750],[0,0,2.6250],[0,0,1.8750],[0,0,1.1250],[0,0,0.3750],[0,0,-0.3750],[0,0,-1.1250],[0,0,-1.8750],[0,0,-2.6250],[0,0,-3.3750],[0,0,-4.1250],[0,0,-4.8750],[0,0,-5.6250],[0,0,-6.3750],[0,0,-7.1250],[0,0,-7.8750],[0,0,-9.3750],[0,0,-10.8750],[0,0,-12.3750],[0,0,-13.8750],[0,0,-15.3750]])

data = aco_in[0:(8*8192),:]
FS = 12000
elev = np.arange(-90,91,1)
az = 0
c = 1435
fft_window = np.hanning(1026)
fft_window = np.delete(fft_window,[0,1025])
overlap = 0.5
NFFT = 1024
f_range = (40,2048)
weighting= 'icex_hanning'

beamform_output,t,flist = beamform_3D(data, p, FS, elev, az, c, f_range, fft_window, NFFT, overlap=0.5, weighting='icex_hanning')

S = beamform_output[:,100,0,:].T
for row in range(S.shape[0]):
    S[row,:] = S[row,:]/np.max(S[row,:])
librosa.display.specshow(10*np.log10(beamform_output[:,100,0,:].T), x_axis='time',y_axis='mel', sr=FS,fmin=40,fmax=2048)
plt.colorbar()
plt.tight_layout()
plt.show()

