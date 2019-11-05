import numpy as np

def chirpz(x,A,W,M):
    """Compute the chirp z-transform.
    The discrete z-transform,
    X(z) = \sum_{n=0}^{N-1} x_n z^{-n}
    is calculated at M points,
    z_k = AW^-k, k = 0,1,...,M-1
    for A and W complex, which gives
    X(z_k) = \sum_{n=0}^{N-1} x_n z_k^{-n}    
    """
    A = np.complex(A)
    W = np.complex(W)
    if np.issubdtype(np.complex,x.dtype) or np.issubdtype(np.float,x.dtype):
        dtype = x.dtype
    else:
        dtype = float

    x = np.asarray(x,dtype=np.complex)
    
    N = x.size
    L = int(2**np.ceil(np.log2(M+N-1)))

    n = np.arange(N,dtype=float)
    y = np.power(A,-n) * np.power(W,n**2 / 2.) * x 
    Y = np.fft.fft(y,L)

    v = np.zeros(L,dtype=np.complex)
    v[:M] = np.power(W,-n[:M]**2/2.)
    v[L-N+1:] = np.power(W,-n[N-1:0:-1]**2/2.)
    V = np.fft.fft(v)
    
    g = np.fft.ifft(V*Y)[:M]
    k = np.arange(M)
    g *= np.power(W,k**2 / 2.)

    return g


def beamform_3D(data, p, FS, elev, az, c, f_range, fft_window, NFFT, overlap=0.5, weighting='hanning'):

	# Define Variables
	N = np.shape(p)[0]
	beam_elev = (90-elev)*(np.pi/180)
	beam_az = az*(np.pi/180)
	win_len = np.shape(fft_window)[0]
	t_end = np.shape(data)[0]/FS

	num_chn = np.shape(data)[1]	

	fft_window = np.tile(fft_window,(num_chn,1))

	# Formate Data
	win_start = np.round(win_len-win_len*overlap)
	num_win = np.round(np.shape(data)[0]/win_start)-1
	beamform_output = np.zeros(num_win,np.shape(beam_elev)[0],np.shape(beam_az)[0],NFFT)
	t = np.zeros(num_win,1)

	# FFT Data
	f1 = f_range[0]
	f2 = f_range[-1]
	w = np.exp(-1j*2*np.pi*(f2-f1)/(NFFT*FS))
	a = np.exp(1j*2*np.pi*f1/FS)

	ts_f_mat = np.zeros(NFFT,N,num_win)
	for l in range(num_win):
		ts_f_mat[:,:,l] = (1/(np.sqrt(FS)*np.linalg.norm(fft_window[:,0])))*np.sqrt(2)*chirpz(fft_window*data[l*win_start-win_start:l*win_start-win_start+win_len-1,:],a,w,NFFT)
		t[l] = ((l+1)*win_start-win_start)/FS;


	# Start Beamforming
	flist = np.linspace(f1,f2,NFFT).T 
	k = 2*np.pi*flist/c

	# linear window
	if weighting == 'uniform':
		win = np.ones((1,N))
		win = win/np.linalg.norm(win)

	# hanning window	
	if weighting == 'hanning':
		win = np.hanning(N)
		win = win/np.linalg.norm(win)

	# icex hanning
	if weighting == 'icex_hanning':
		win = np.hanning(42)
		win = np.delete(win,[1,3,5,7,9,40,38,36,34,32])
		win = win/np.linalg.norm(win)

	# build steering vectors
	for j in beam_az:
		for mm in beam_elev:
			steer = np.exp(1j * k *(np.sin(mm)*np.cos(j)*p[:,1].T+np.sin(mm)*np.sin(j)*p[:,2].T+np.cos(mm)*p[:,3].T))

			#apply weighting
			steer = steer*(np.ones(np.shape(k)[0],1)*win)

			#beamform
			for l in range(num_win):
				b_elem = sum((np.conj(steer)*ts_f_mat[:,:,l]).T);
				beamform_output[l,mm,j,:] = np.abs(b_elem)**2;

	t = t - t(1);
	t_end = t_end - t(1);






