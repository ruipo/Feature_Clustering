import os
curdir = '/Users/Rui/Documents/Graduate/Research/Feature_Clustering/whale_detection/PIFSC/'
os.chdir(curdir)
import matplotlib.pyplot as plt 
import matplotlib.dates as mds
from save_obj import read_object
import numpy as np
import copy

wc_startt_act = np.loadtxt(curdir+'wc_starttimes.txt')

path = '/Users/Rui/Documents/Graduate/Research/Feature_Clustering/whale_detection/PIFSC/Features_mt50/'

directory = [f for f in os.listdir(path) if f.endswith(".pkl")]

wc_startt = []

for ff in directory:
	Feature = read_object(path+ff)
	if Feature.end_t-Feature.start_t>1.25:
		wc_startt.append(Feature.start_t)
	else:
		continue

wc_startt = np.round(np.sort(wc_startt))

falsepos_list = copy.deepcopy(wc_startt).tolist()
falseneg_list = copy.deepcopy(wc_startt_act).tolist()
for t1 in wc_startt:
	diff = np.abs(wc_startt_act-t1)
	if np.min(diff)<=2 and t1 in falsepos_list and wc_startt_act[np.argmin(diff)] in falseneg_list:
		falsepos_list.remove(t1)
		falseneg_list.remove(wc_startt_act[np.argmin(diff)])
	else:
		continue

tp = wc_startt_act.shape[0]-len(falseneg_list)
fn = len(falseneg_list)
fp = len(falsepos_list)
tn = wc_startt.shape[0]-len(falsepos_list)

TPR= tp/(tp+fn)
FPR = fp/(fp+tn)

print('true_positive_rate: ', TPR)
print('false_positive_rate: ',  FPR)



