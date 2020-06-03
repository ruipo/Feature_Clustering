
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mds
from scipy.interpolate import interp1d

def get_array_pitch(Path,figshow=False):

	time = pd.read_csv(Path+'DB_TIME.klog',delimiter=' ')
	time = time.values
	time = time[~pd.isna(time)]
	uptime = time[0::4]
	eptime = np.array([float(ii) for ii in time[3::4]])

	up2ep = interp1d(uptime,eptime)

	ael_pitch = pd.read_csv(Path+'AEL_PITCH_C.klog',delimiter=' ',low_memory=False)
	ael_pitch = ael_pitch.values
	ael_pitch = ael_pitch[~pd.isna(ael_pitch)]
	uptime_ael = ael_pitch[0::4]
	uptime_ael = uptime_ael[uptime_ael<=uptime[-1]]

	ael_eptime = np.array([up2ep(ii) for ii in uptime_ael])
	ael_pitch = np.array([float(ii) for ii in ael_pitch[3::4]])
	ael_pitch = ael_pitch[0:uptime_ael.shape[0]]

	ep2pitch = interp1d(ael_eptime, ael_pitch)

	ael_utc = mds.epoch2num(ael_eptime)

	if figshow:
		fig = plt.figure()
		plt.plot(ael_utc,ael_pitch)
		ax = plt.gca()
		ax.xaxis.set_major_formatter(formatter=mds.DateFormatter('%H:%M:%S'))
		plt.xlabel('Time')
		plt.ylabel('Array Pitch (Deg.)')
		fig.autofmt_xdate()
		plt.show()

	return(ep2pitch)