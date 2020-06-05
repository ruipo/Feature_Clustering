############################### Imports ##################################################
import os
curdir = '/Users/Rui/Documents/Graduate/Research/Feature_Clustering/dev_new/'
os.chdir(curdir)
import matplotlib.pyplot as plt 
import matplotlib.dates as mds
from save_obj import read_object
import numpy as np

path = '/Users/Rui/Documents/Graduate/Research/Feature_Clustering/dev_new/Features/'

directory = [f for f in os.listdir(path) if f.endswith(".pkl")]

elevmax_list = []
f_diff_list = []
t_diff_list = []
t_start_list = []
t_end_list = []
f_start_list = []
f_end_list = []
orientation_list = []
tbe_start_t = []

endf_below_160hz=0
endf_160_320hz=0
endf_320_640hz=0
endf_640_1280hz=0
endf_1280_2048hz=0
st_bb_list = []
st_nb_list = []
lt_bb_list = []
lt_nb_list = []
stnb_tdiff_list = []
ltbb_tdiff_list = []
stbb_tdiff_list = []
ltnb_tdiff_list = []
stnb_fdiff_list = []
ltbb_fdiff_list = []
stbb_fdiff_list = []
ltnb_fdiff_list = []
stnb_tstart_list = []
stbb_tstart_list = []
ltnb_tstart_list = []
ltbb_tstart_list = []
stnb_tbe = []
stbb_tbe = []
ltnb_tbe = []
ltbb_tbe = []
etype_list = []

for ff in directory:

	Feature = read_object(path+ff)

	if Feature.end_f < 160:
		endf_below_160hz = endf_below_160hz+1
	elif 160<= Feature.end_f <320:
		endf_160_320hz = endf_160_320hz+1
	elif 320<= Feature.end_f <640:
		endf_320_640hz = endf_320_640hz+1
	elif 640<= Feature.end_f <1280:
		endf_640_1280hz = endf_640_1280hz+1
	else:
		endf_1280_2048hz = endf_1280_2048hz+1


	if (Feature.end_f-Feature.start_f)>=500 and (Feature.end_t-Feature.start_t)<=5:
		st_bb_list.append(Feature)#short-time-broad-band
		stnb_tstart_list.append(Feature.start_t)
		stbb_tdiff_list.append(Feature.end_t-Feature.start_t)
		stbb_fdiff_list.append(Feature.end_f-Feature.start_f)
		etype = 2
	elif (Feature.end_f-Feature.start_f)<500 and (Feature.end_t-Feature.start_t)>5:
		lt_nb_list.append(Feature) #long-time-narrow-band
		stbb_tstart_list.append(Feature.start_t)
		ltnb_tdiff_list.append(Feature.end_t-Feature.start_t)
		ltnb_fdiff_list.append(Feature.end_f-Feature.start_f)
		etype = 3
	elif (Feature.end_f-Feature.start_f)<500 and (Feature.end_t-Feature.start_t)<=5:
		st_nb_list.append(Feature) #short-time-narrow-band
		ltnb_tstart_list.append(Feature.start_t)
		stnb_tdiff_list.append(Feature.end_t-Feature.start_t)
		stnb_fdiff_list.append(Feature.end_f-Feature.start_f)
		etype = 1
	else:
		lt_bb_list.append(Feature) #long-time-broad-band
		ltbb_tstart_list.append(Feature.start_t)
		ltbb_tdiff_list.append(Feature.end_t-Feature.start_t)
		ltbb_fdiff_list.append(Feature.end_f-Feature.start_f)
		etype = 4

	etype_list.append(etype)
	elevmax_list.append(Feature.elevmax)
	t_start_list.append(Feature.start_t)
	t_end_list.append(Feature.end_t)
	f_start_list.append(Feature.start_f)
	f_end_list.append(Feature.end_f)
	orientation_list.append(Feature.orientation)
	f_diff_list.append(Feature.end_f-Feature.start_f)
	t_diff_list.append(Feature.end_t-Feature.start_t)

print('percent of events with f content below 160hz: ', endf_below_160hz/len(f_end_list)*100)
print('percent of events with f content 160-320hz: ', endf_160_320hz/len(f_end_list)*100)
print('percent of events with f content 320-640hz: ', endf_320_640hz/len(f_end_list)*100)
print('percent of events with f content 640-1280hz: ', endf_640_1280hz/len(f_end_list)*100)
print('percent of events with f content 1280-2048hz: ', endf_1280_2048hz/len(f_end_list)*100)
print('percent of stbb events: ',len(st_bb_list)/len(orientation_list)*100)
print('percent of ltbb events: ',len(lt_bb_list)/len(orientation_list)*100)
print('percent of stnb events: ',len(st_nb_list)/len(orientation_list)*100)
print('percent of ltnb events: ',len(lt_nb_list)/len(orientation_list)*100)

t_start_list = np.sort(t_start_list)
for ts in range(len(t_start_list[0:-1])):
	tbe_start_t.append(t_start_list[ts+1]-t_start_list[ts])

stnb_tstart_list = np.sort(stnb_tstart_list)
for ts in range(len(stnb_tstart_list[0:-1])):
	stnb_tbe.append(stnb_tstart_list[ts+1]-stnb_tstart_list[ts])

stbb_tstart_list = np.sort(stbb_tstart_list)
for ts in range(len(stbb_tstart_list[0:-1])):
	stbb_tbe.append(stbb_tstart_list[ts+1]-stbb_tstart_list[ts])

ltnb_tstart_list = np.sort(ltnb_tstart_list)
for ts in range(len(ltnb_tstart_list[0:-1])):
	ltnb_tbe.append(ltnb_tstart_list[ts+1]-ltnb_tstart_list[ts])

ltbb_tstart_list = np.sort(ltbb_tstart_list)
for ts in range(len(ltbb_tstart_list[0:-1])):
	ltbb_tbe.append(ltbb_tstart_list[ts+1]-ltbb_tstart_list[ts])

# plot event type with time
fig = plt.figure(figsize=(20,8))
plt.plot(mds.epoch2num(t_start_list),etype_list,'.')
ax = plt.gca()
ax.xaxis.set_major_formatter(formatter=mds.DateFormatter('%H:%M:%S'))
plt.xlabel('Event Start Time (Mar. 13, 2016 UTC)',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks((1,2,3,4),('stnb', 'stbb','ltnb', 'ltbb'),fontsize=15)
plt.ylim(0,5)
plt.ylabel('Event Type',fontsize=15)
plt.grid()
fig.autofmt_xdate()
#plt.show()
plt.savefig(curdir+'figures/event_type.png')
plt.clf()
plt.close()


# Plot event duration
fig1 = plt.figure(figsize=(20,8))
plt.subplot(1,2,1,autoscale_on=True)
plt.hist(t_diff_list,bins=150,histtype='bar',color='green', edgecolor='black')
plt.xlabel('Event Duration (Sec)',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel('Count',fontsize=15)
plt.yscale('log')
plt.grid()

plt.subplot(1,2,2,autoscale_on=True)
plt.hist(t_diff_list,bins=150,histtype='bar',color='green', edgecolor='black',density=1,cumulative=1)
plt.xlabel('Event Duration (Sec)',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel('Cumulative Density',fontsize=15)
plt.grid()

plt.savefig(curdir+'figures/event_duration.png')
plt.clf()
plt.close()

# Plot event bandwdith
fig1 = plt.figure(figsize=(20,8))
plt.subplot(1,2,1,autoscale_on=True)
plt.hist(f_diff_list,bins=150,histtype='bar',color='blue', edgecolor='black')
plt.xlabel('Frequency Bandwidth (Hz)',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel('Count',fontsize=15)
plt.yscale('log')
plt.grid()

plt.subplot(1,2,2,autoscale_on=True)
plt.hist(f_diff_list,bins=150,histtype='bar',color='blue', edgecolor='black',density=1,cumulative=1)
plt.xlabel('Frequency Bandwidth (Hz)',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel('Cumulative Density',fontsize=15)
plt.grid()

plt.savefig(curdir+'figures/event_bandwidth.png')
plt.clf()
plt.close()

# plot event max_elev
fig = plt.figure(figsize=(20,8))
plt.plot(mds.epoch2num(t_start_list),elevmax_list,'.')
ax = plt.gca()
ax.xaxis.set_major_formatter(formatter=mds.DateFormatter('%H:%M:%S'))
plt.xlabel('Event Start Time (Mar. 13, 2016 UTC)',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.ylim(-45,45)
plt.ylabel('Max Beamform Elevation (Deg.)',fontsize=15)
plt.grid()
fig.autofmt_xdate()
plt.savefig(curdir+'figures/event_maxelev.png')
plt.clf()
plt.close()

# plot tbe starttime
fig1 = plt.figure(figsize=(20,8))
plt.subplot(1,2,1,autoscale_on=True)
plt.hist(tbe_start_t,bins=150,histtype='bar',color='green', edgecolor='black')
plt.xlabel('Time Between Events (Sec)',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel('Count',fontsize=15)
plt.yscale('log')
plt.grid()

plt.subplot(1,2,2,autoscale_on=True)
plt.hist(tbe_start_t,bins=150,histtype='bar',color='green', edgecolor='black',density=1,cumulative=1)
plt.xlabel('Time Between Events (Sec)',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel('Cumulative Density',fontsize=15)
plt.grid()

plt.savefig(curdir+'figures/tbe_startt.png')
plt.clf()
plt.close()


# plot event duration
fig1 = plt.figure(figsize=(20,8))
plt.subplot(1,2,1,autoscale_on=True)
plt.hist([stnb_tdiff_list,stbb_tdiff_list,ltnb_tdiff_list,ltbb_tdiff_list],bins=150,histtype='bar',color=['green','blue','red','orange'], edgecolor='black',stacked=1)
plt.xlabel('Event Duation (Sec)',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel('Count',fontsize=15)
plt.yscale('log')
plt.legend(('stnb', 'stbb', 'ltnb', 'ltbb'))
plt.grid()

plt.subplot(1,2,2,autoscale_on=True)
plt.hist([stnb_tdiff_list,stbb_tdiff_list,ltnb_tdiff_list,ltbb_tdiff_list],bins=150,histtype='bar',color=['green','blue','red','orange'], edgecolor='black',density=1,cumulative=1,stacked=1)
plt.xlabel('Event Duration (Sec)',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel('Cumulative Density',fontsize=15)
plt.legend(('stnb', 'stbb', 'ltnb', 'ltbb'))
plt.grid()

plt.savefig(curdir+'figures/event_druation_stacked.png')
plt.clf()
plt.close()

# plot event bandwidth
fig1 = plt.figure(figsize=(20,8))
plt.subplot(1,2,1,autoscale_on=True)
plt.hist([stnb_fdiff_list,stbb_fdiff_list,ltnb_fdiff_list,ltbb_fdiff_list],bins=150,histtype='bar',color=['green','blue','red','orange'], edgecolor='black',stacked=1)
plt.xlabel('Frequency Bandwidth (Hz)',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel('Count',fontsize=15)
plt.yscale('log')
plt.legend(('stnb', 'stbb', 'ltnb', 'ltbb'))
plt.grid()

plt.subplot(1,2,2,autoscale_on=True)
plt.hist([stnb_fdiff_list,stbb_fdiff_list,ltnb_fdiff_list,ltbb_fdiff_list],bins=150,histtype='bar',color=['green','blue','red','orange'], edgecolor='black',density=1,cumulative=1,stacked=1)
plt.xlabel('Frequency Bandwidth (Hz)',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel('Cumulative Density',fontsize=15)
plt.legend(('stnb', 'stbb', 'ltnb', 'ltbb'))
plt.grid()

plt.savefig(curdir+'figures/event_bandwidth_stacked.png')
plt.clf()
plt.close()

# plot tbe starttime
fig1 = plt.figure(figsize=(20,8))
plt.subplot(1,2,1,autoscale_on=True)
plt.hist([stnb_tbe,stbb_tbe,ltnb_tbe,ltbb_tbe],bins=150,histtype='bar',color=['green','blue','red','orange'], edgecolor='black',stacked=1)
plt.xlabel('Time Between Events (Sec)',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel('Count',fontsize=15)
plt.yscale('log')
plt.xlim([0,500])
plt.legend(('stnb', 'stbb', 'ltnb', 'ltbb'))
plt.grid()

plt.subplot(1,2,2,autoscale_on=True)
plt.hist([stnb_tbe,stbb_tbe,ltnb_tbe,ltbb_tbe],bins=150,histtype='bar',color=['green','blue','red','orange'], edgecolor='black',density=1,cumulative=1,stacked=1)
plt.xlabel('Time Between Events (Sec)',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel('Cumulative Density',fontsize=15)
plt.legend(('stnb', 'stbb', 'ltnb', 'ltbb'))
plt.xlim([0,500])
plt.grid()

plt.savefig(curdir+'figures/event_tbe_stacked.png')
plt.clf()
plt.close()