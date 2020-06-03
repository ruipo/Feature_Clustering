import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import radon
import librosa

FS = 12000 
n_fft = 4096
hop_length = int(n_fft/2)

flist = librosa.core.mel_frequencies(n_mels=128, fmin=0.0, fmax=2048, htk=False)


path_out = '/Users/Rui/Documents/Graduate/Research/ICEX:SIMI/lstm_eSelect/output/'
directory_out = [f for f in os.listdir(path_out) if f.endswith("output.txt")]
directory_out_sorted = sorted(directory_out,key=lambda x: int(os.path.splitext(x)[0][0:-9]))
#binary = np.loadtxt(path_out+directory_out[10])
output_data_tot = np.array(np.zeros((128,1)))

for ff in directory_out_sorted[18:19]:
	temp = np.loadtxt(path_out+ff)
	output_data_tot = np.concatenate((output_data_tot,temp),axis=1)

output_data_tot = np.delete(output_data_tot,np.s_[0],1)

num_nfft_tot = int(np.ceil(30*FS/hop_length))
tlist = (1/FS)*np.linspace(0,30*FS,num_nfft_tot)

# plt.imshow(np.flipud(output_data_tot))
# plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5))

ax1.set_title("Original")
ax1.imshow(np.flipud(output_data_tot[0:128,:]), cmap=plt.cm.Greys_r,aspect='auto',extent=[tlist[0],tlist[-1],flist[0],flist[-1]])
ax1.plot(tlist[88],flist[64],'*')
theta = np.linspace(0., 180., 180, endpoint=False)
sinogram = radon(np.flipud(output_data_tot[0:128,:]), theta=theta, circle=True)
ax2.set_title("Radon transform\n(Sinogram)")
ax2.set_xlabel("Projection angle (deg)")
ax2.set_ylabel("Projection position (pixels)")
ax2.imshow(sinogram, cmap=plt.cm.Greys_r,
           extent=(0, 180,sinogram.shape[0]/2,-sinogram.shape[0]/2), aspect='auto')

fig.tight_layout()
plt.show()