import os
curdir = '/Users/Rui/Documents/Graduate/Research/Feature_Clustering/whale_detection/PIFSC/'
os.chdir(curdir)
import matplotlib.pyplot as plt 
import matplotlib.dates as mds
from save_obj import read_object
import numpy as np
import copy

wc_startt_act = np.loadtxt(curdir+'wc_starttimes.txt')

path = '/Users/Rui/Documents/Graduate/Research/Feature_Clustering/whale_detection/PIFSC/Features/'

directory = [f for f in os.listdir(path) if f.endswith(".pkl")]

wc_startt = []
st_startt = []

for ff in directory:
	Feature = read_object(path+ff)
	if Feature.type == 'wc':
	#if Feature.end_t-Feature.start_t>1.25:
		wc_startt.append(Feature.start_t)
	else:
		st_startt.append(Feature.start_t)

wc_startt = np.round(np.sort(wc_startt))
st_startt = np.round(np.sort(st_startt))

falsepos_list = copy.deepcopy(wc_startt).tolist()
falseneg_list = copy.deepcopy(wc_startt_act).tolist()
for t1 in wc_startt:
	diff = np.abs(wc_startt_act-t1)
	if np.min(diff)<=1 and wc_startt_act[np.argmin(diff)] in falseneg_list:
		falsepos_list.remove(t1)
		falseneg_list.remove(wc_startt_act[np.argmin(diff)])
	else:
		continue

trueneg_list = copy.deepcopy(st_startt).tolist()
for t2 in st_startt:
	diff = np.abs(wc_startt_act-t2)
	if np.min(diff)<=1 and wc_startt_act[np.argmin(diff)] in trueneg_list:
		trueneg_list.remove(t2)
	else:
		continue

tp = wc_startt_act.shape[0]-len(falseneg_list)
fn = len(falseneg_list)
fp = len(falsepos_list)
#tn = wc_startt.shape[0]-len(falsepos_list)
tn = len(trueneg_list)

TPR= tp/(tp+fn)
FPR = fp/(fp+tn)

print('true_positive_rate: ', TPR)
print('false_positive_rate: ',  FPR)


plt.plot(wc_startt,np.ones(len(wc_startt)),'b*')
plt.plot(wc_startt_act,1.1*np.ones(len(wc_startt_act)),'r*')
plt.ylim(0,10)
plt.grid(True)
plt.show()
