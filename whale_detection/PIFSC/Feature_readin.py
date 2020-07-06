import os
curdir = '/Users/Rui/Documents/Graduate/Research/Feature_Clustering/whale_detection/PIFSC/'
os.chdir(curdir)
import matplotlib.pyplot as plt 
import matplotlib.dates as mds
from save_obj import read_object
import numpy as np
import copy

wc_startt_act = np.loadtxt(curdir+'wc_starttimes.txt')

path = '/Users/Rui/Documents/Graduate/Research/Feature_Clustering/whale_detection/PIFSC/Features_mt30/'

directory = [f for f in os.listdir(path) if f.endswith(".pkl")]

wc_startt = []
st_startt = []

for ff in directory:
	Feature = read_object(path+ff)
	#if Feature.type == 'wc':
	if Feature.end_t-Feature.start_t>0:
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



fpr = [0.001,	0.0945,	0.139,	0.215,	1]
tpr = [0.001,	0.633,	0.709,	0.735,	1]
plt.plot(fpr,tpr,'b.')
x = np.linspace(0.000001,1,10000)
y = 0.14269*np.log(x)+0.9802
plt.plot(x,y,'b-')
plt.grid(True)
plt.xlim(0,1)
plt.ylim(0,1)
plt.xlabel('False Positive Rate',fontsize=20)
plt.ylabel('True Positive Rate',fontsize=20)
plt.title('ROC Curve [AUG = 0.84]',fontsize=20)
plt.show()