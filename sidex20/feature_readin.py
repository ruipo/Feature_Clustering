import os
curdir = '/Users/Rui/Documents/Graduate/Research/Feature_Clustering/sidex20/'
os.chdir(curdir)
import matplotlib.pyplot as plt 
import matplotlib.dates as mds
from save_obj import read_object
import numpy as np
import copy

st_startt_act = np.loadtxt(curdir+'calib_event_eptimes.txt')
st_startt_act = st_startt_act-3

path = '/Users/Rui/Documents/Graduate/Research/Feature_Clustering/sidex20/Features/'

directory = [f for f in os.listdir(path) if f.endswith(".pkl")]

st_startt = []
lt_startt = []

for ff in directory:
	Feature = read_object(path+ff)
	#if Feature.type == 'st':
	if Feature.end_t-Feature.start_t<=2:
		st_startt.append(Feature.start_t)
	else:
		lt_startt.append(Feature.start_t)

st_startt = np.round(np.sort(st_startt))
lt_startt = np.round(np.sort(lt_startt))

falsepos_list = copy.deepcopy(st_startt).tolist()
falseneg_list = copy.deepcopy(st_startt_act).tolist()
for t1 in st_startt:
	diff = np.abs(st_startt_act-t1)
	if np.min(diff)<=2 and st_startt_act[np.argmin(diff)] in falseneg_list:
		falsepos_list.remove(t1)
		falseneg_list.remove(st_startt_act[np.argmin(diff)])
	else:
		continue

trueneg_list = copy.deepcopy(st_startt).tolist()
for t2 in lt_startt:
	diff = np.abs(st_startt_act-t2)
	if np.min(diff)<=2 and st_startt_act[np.argmin(diff)] in trueneg_list:
		trueneg_list.remove(t2)
	else:
		continue

tp = st_startt_act.shape[0]-len(falseneg_list)
fn = len(falseneg_list)
fp = len(falsepos_list)
#tn = wc_startt.shape[0]-len(falsepos_list)
tn = len(trueneg_list)

TPR= tp/(tp+fn)
FPR = fp/(fp+tn)

print('true_positive_rate: ', TPR)
print('false_positive_rate: ',  FPR)


plt.plot(st_startt,np.ones(len(st_startt)),'b*')
plt.plot(st_startt_act,1.1*np.ones(len(st_startt_act)),'r*')
plt.ylim(0,10)
plt.grid(True)
plt.show()



fpr = [0.159,	0.201,	0.217,	0.317,	1]
tpr = [0.66,	0.713,	0.739,	0.771,	1]
plt.plot(fpr,tpr,'b.')
x = np.linspace(0.000001,1,10000)
y = 0.1396*np.log(x)+0.9504
plt.plot(x,y,'b-')
plt.grid(True)
plt.xlim(0,1)
plt.ylim(0,1)
plt.xlabel('False Positive Rate',fontsize=20)
plt.ylabel('True Positive Rate',fontsize=20)
plt.title('ROC Curve [AUG = 0.81]',fontsize=20)
plt.show()