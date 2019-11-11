import numpy as np

def chirpz(x,k,w,a):
    """Compute the chirp z-transform.
    The discrete z-transform,
    X(z) = \sum_{n=0}^{N-1} x_n z^{-n}
    is calculated at M points,
    z_k = AW^-k, k = 0,1,...,M-1
    for A and W complex, which gives
    X(z_k) = \sum_{n=0}^{N-1} x_n z_k^{-n}  

    x = [timestep,chn]  
    """
    a = np.complex(a)
    w = np.complex(w)
    if np.issubdtype(np.complex,x.dtype) or np.issubdtype(np.float,x.dtype):
        dtype = x.dtype
    else:
        dtype = float

    x = np.asarray(x,dtype=np.complex)

    m,n = x.shape
    nfft = int(2**np.ceil(np.log2(m+n-1)))

    kk = np.arange((-m+1),np.max([k,m]),1)
    kk2 = (kk**2)/2
    ww = w**(kk2)
    nn = np.arange(m)
    aa = a**(-nn)
    aa = aa*ww[m+nn-1]
    aa = np.expand_dims(aa,axis=1)
    z = np.repeat(aa,n,axis=1)
    y = x*z

    fy = np.fft.fft(y,nfft,axis=0)
    fv = np.fft.fft(1/ww[0:(k-1+m)],nfft,axis=0)
    fv = np.expand_dims(fv,axis=1)
    fvz = np.repeat(fv,n,axis=1)
    fyz = fy*fvz
    g = np.fft.ifft(fyz,axis=0)

    ww = np.expand_dims(ww,axis=1)
    wwz = ww[m-1:m+k-1,:]
    wwz = np.repeat(wwz,n,axis=1)
    g = g[m-1:m+k-1,:]*wwz

    return g

def beamform_3D(data, p, FS, elev, az, c, f_range, fft_window, NFFT, overlap=0.5, weighting='hanning'):

	# Define Variables
	N = int(np.shape(p)[0])
	beam_elev = (90-elev)*(np.pi/180)
	beam_az = az*(np.pi/180)
	win_len = int(np.shape(fft_window)[0])
	t_end = np.shape(data)[0]/FS

	fft_window = np.tile(fft_window,(N,1)).T

	if len(np.shape(beam_elev)) == 0:
		num_elev = 1
		beam_elev = np.array([beam_elev])
	else:
		num_elev = np.shape(beam_elev)[0]

	if len(np.shape(beam_az)) == 0:
		num_az = 1
		beam_az = np.array([beam_az])
	else:
		num_az = np.shape(beam_az)[0]

	# Formate Data
	win_start = int(np.round(win_len-win_len*overlap))
	num_win = int(np.round(np.shape(data)[0]/win_start))
	beamform_output = np.zeros((num_win,num_elev,num_az,NFFT))
	t = np.zeros((num_win,1))

	# FFT Data
	f1 = f_range[0]
	f2 = f_range[-1]
	w = np.exp(-1j*2*np.pi*(f2-f1)/(NFFT*FS))
	a = np.exp(1j*2*np.pi*f1/FS)

	ts_f_mat = np.zeros((NFFT,N,num_win),dtype=complex)
	for l in range(num_win):

		if fft_window.shape[0] != np.shape(data[l*win_start:l*win_start+win_len,:])[0]:
			diff = fft_window.shape[0] - np.shape(data[l*win_start:l*win_start+win_len,:])[0]
			data = np.vstack((data,np.zeros((diff,N))))

		ts_f_mat[:,:,l] = (1/(np.sqrt(FS)*np.linalg.norm(fft_window[:,0])))*np.sqrt(2)*chirpz(fft_window*data[l*win_start:l*win_start+win_len,:],NFFT,w,a)
		t[l] = (l*win_start+win_len/2)/FS


	# Start Beamforming
	flist = np.linspace(f1,f2,NFFT)
	k = 2*np.pi*flist/c
	k = np.expand_dims(k, axis=1)

	# linear window
	if weighting == 'uniform':
		win = np.ones((N,))
		win = win/np.linalg.norm(win)
		win = np.expand_dims(win,axis = 0)

	# hanning window	
	if weighting == 'hanning':
		win = np.hanning(N+2)
		win = np.delete(win,[0,N+1])
		win = win/np.linalg.norm(win)
		win = np.expand_dims(win,axis = 0)

	# icex hanning
	if weighting == 'icex_hanning':
		win = np.hanning(44)
		win = np.delete(win,[0,2,4,6,8,10,43,41,39,37,35,33])
		win = win/np.linalg.norm(win)
		win = np.expand_dims(win,axis = 0)

	# build steering vectors
	for j in range(int(num_az)):
		for mm in range(int(num_elev)):

			s = np.sin(beam_elev[mm])*np.cos(beam_az[j])*p[:,0]+np.sin(beam_elev[mm])*np.sin(beam_az[j])*p[:,1]+np.cos(beam_elev[mm])*p[:,2]
			s = np.expand_dims(s,axis=0)
			steer = np.exp(1j * np.matmul(k,s))

			#apply weighting
			steer = steer*(np.matmul(np.ones((np.shape(k)[0],1)),win))

			#beamform
			for l in range(num_win):
				
				b_elem = np.sum((np.conj(steer)*ts_f_mat[:,:,l]),axis = 1)
				beamform_output[l,mm,j,:] = np.abs(b_elem)**2

	return beamform_output,t,flist







