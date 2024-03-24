#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 16:31:46 2022

@author: pdhara
"""
import os
import numpy as np
import scipy as sc
import pandas as pd
import matplotlib.pyplot as plt
import sys, math
import mne, mne_nirs

sys.path.insert(1, '/data2/pdhara/decoding_pire/iaoToolboxCustom/')
import proc #helpers, io, proc 
import nirs_help
import supporting_func, ampd
from proc import fNIRSreadParams, FilterParams #, EpochProps
import help_functions_stress    

info=fNIRSreadParams(
	preload=True,
	verbose=None)

filterParams = FilterParams(
	l_freq=1,
	h_freq=1.9,
	method='fir',
	fir_design='firwin',
	phase='zero-double')

path_dict={'06_sub_maryam': ['2021-10-06_003', 'log_data', [(27.90, 142.47), (4630.59, 4732.28)]],
			'07_sub_Yichi': ['2021-10-25_001', 'log_data', [(15.39, 161.10), (3700.05, 3825.94)]],
			'08_sub_Chinmaya': ['2021-10-26_002', 'log_data', [(25.22, 112.73),	(4377.42, 4456.86)]],
			#'09_sub_Katharina': ['2021-10-28_002', 'log_data', [(52.87, 127.97),	(4072.61, 4145.96)]],
			'10_sub_Martin': ['2021-11-02_003', 'log_data', [(46.80, 120.55),	(4351.66, 4474.44)]],
			'11_sub_Sabine': ['2021-11-05_002', 'log_data', [(30.44, 152.23),	(4687.38, 4832.41)]],
			'12_sub_Mohammad': ['2021-11-08_001', 'log_data', [(102.03, 222.54),	(5005.38, 5130.31)]],
			'13_sub_Gizem': ['2021-11-09_001', 'log_data', [(43.77, 191.85),	(4302.88, 4526.49)]],
			'14_sub_Shaheel': ['2021-11-11_001', 'log_data', [(82.07, 205.04),	(4901.64, 5023.67)]],
            '15_sub_Nils': ['2021-11-18_001', 'log_data', [(63.29, 220.40),	(3851.89, 3983.02)]]
			}
## Grouping of channels for stress precition
SD_list=[[(2, 1), (3, 1), (2, 5), (6, 5), (7, 5)],
		[(1, 1), (1, 2), (1, 3), (3, 3), (4, 3), (3, 6), (4, 7), (7, 6), (8, 6), (8, 3), (8, 7), (9, 7)],
		[(5, 2), (5, 8), (4, 2), (9, 8), (10, 8), (10, 9)]]
grp_ch_names_hbo=help_functions_stress.flatten_list(nirs_help.create_grouping_optodes(SD_list, oxy_type='hbo')) # convert source detector pairs into channel name e.g., S1_D1 hbo
grp_ch_names_hbr=help_functions_stress.flatten_list(nirs_help.create_grouping_optodes(SD_list, oxy_type='hbr'))
grp_ch_names=np.append(grp_ch_names_hbo, grp_ch_names_hbr)
# End of Grouping of channels ...
#%%
baseFolder='/data2/pdhara/work/data/pire_data'

for sub, value in path_dict.items():
    break
# Reading of fNIRS and behavioral data
data_path=os.path.join(baseFolder, sub, value[0])
fnirs_data=proc.load_fNIRS_data(data_path, info)
### converting raw intensity to optical intensity
fnirs_data_od=mne.preprocessing.nirs.optical_density(fnirs_data)
### Converting from optical density to haemoglobin
fnirs_data_haemo=mne.preprocessing.nirs.beer_lambert_law(fnirs_data_od)
raw_resampled = fnirs_data_haemo.resample(10)
## Removing heart rate from signal
fnirs_data_filt=proc.filter(raw_resampled.copy(), filterParams, 'bandpass')
#%%
# take only single channel for testing the coding
fnirs_1_raw_data=raw_resampled.copy().pick_channels(['S1_D1 hbo']).get_data().flatten()
fnirs_1_ch=fnirs_data_filt.copy().pick_channels(['S1_D1 hbo'])
fnirs_1_data=fnirs_1_ch.get_data().flatten()

fig, ax =plt.subplots(nrows=2, ncols=1, sharex=True)
ax[0].plot(fnirs_1_raw_data)
ax[0].grid()
ax[1].plot(fnirs_1_data)
ax[1].grid()
plt.show()

# Sampling interval
T=1/fnirs_data_filt.info['sfreq'] # sampling period
#%%
def plot_timeseries_marker(x1, y1, x_m, y_m, title_name="Plot"):
    '''
    Plot the actual time series (x1, y1) and detected peak points (x_m, y_m)

    Parameters
    ----------
    x1 : TYPE
        sample number for actual time series data
    y1 : TYPE
        value at each sample
    x_m : TYPE
        Sample number where peaks are detected
    y_m : TYPE
        Value at the peak points

    Returns
    -------
    None.

    '''
    fig, ax=plt.subplots()
    ax.plot(x1, y1)
    ax.plot(x_m, y_m, "ro")
    plt.title(title_name)
    plt.show()
    return
#%%
### Detect peaks using AMPD algorithm. pks holds the peak index
debug=True
if debug:
    pks_ampd, ZLSM, l_scale=ampd.find_peaks_original(fnirs_1_data, debug=True)
else:
    pks_ampd=ampd.find_peaks_original(fnirs_1_data)

# Plot the time series data along with red dots at peak position
idx=np.arange(0, len(fnirs_1_data))
plot_timeseries_marker(idx, fnirs_1_data, pks_ampd, fnirs_1_data[pks_ampd], "Detected peaks: AMPD")
# End of Detect Peaks using AMPD ...
#%% Compute first difference of time series and its first difference
first_diff=help_functions_stress.first_difference_series(fnirs_1_data, T)
max_first_diff=np.max(first_diff)

#%% This function is written to plot the raw data, computed function (m2d, s1, s2 etc) on raw data, and first difference 
# function of raw data. This function is particularly for debugging
def plot_raw_fn_diff(a, b, c, text1="Raw Data", text2="S1 Function", text3="First Difference"):
    fig, ax=plt.subplots(nrows=3, ncols=1, sharex=True)
    ax[0].plot(a), ax[0].title.set_text(text1), ax[0].grid()
    ax[1].plot(b), ax[1].title.set_text(text2), ax[1].grid()
    ax[2].plot(c), ax[2].title.set_text(text3), ax[2].grid()
    plt.show()
    return

#%%
# Detect peaks using M2D algorithm
start=3
m2d_values=np.zeros(np.size(fnirs_1_data))
for i in range(start, np.size(fnirs_1_data)-start):
    m2d_values[i]=help_functions_stress.m2d(fnirs_1_data, i)

## correction in M2D algorithm
# Use thresholding approach to detect the index of peaks
threshold_m2d=100*np.mean(m2d_values)+ np.std(m2d_values)
#pks_m2dd_threshold=np.argwhere(m2d_values>threshold_m2d)
pks_m2d_threshold=np.intersect1d(np.where(m2d_values>=threshold_m2d)[0], np.where(first_diff>=0.1*max_first_diff)[0]) 

# Plot raw data, and computed m2d_values and first difference
plot_raw_fn_diff(fnirs_1_data, m2d_values, first_diff, "Raw Data", "M2D values", "First Difference")
'''fig, ax =plt.subplots(nrows=3, ncols=1, sharex=True)
ax[0].plot(fnirs_1_data); ax[0].grid()
ax[1].plot(m2d_values); ax[1].grid()
ax[2].plot(first_diff); ax[2].grid()
plt.show() '''

# Use peak detection method 
pks_m2d_peaks, _=sc.signal.find_peaks(m2d_values, distance=12) # distance=15   rel_height=1e-8

pks_m2d=pks_m2d_peaks
plot_timeseries_marker(idx, fnirs_1_data, pks_m2d, fnirs_1_data[pks_m2d], "Detected peaks: M2D")
# End of Detect peaks using M2D algorithm
#%% Create a sine wave to explain s1 function
fs=10
f=1
duration=4
num_samples=fs*duration
t=np.linspace(0, duration, num_samples)
y=np.sin(2*np.pi*f*t)
fig=plt.figure()
plt.plot(t, y)
plt.show()

def s1_func_test(k, i, x_i, y):
    N_minus_max=max(x_i-y[i-k:i])
    N_plus_max=max(x_i-y[i+1:i+k+1])
    # print("i=", i, "x_i=", x_i, "x_i-y[i-k:i]=", x_i-y[i-k:i], "N_minus_max=", N_minus_max)
    # print("x_i-y[i+1:i+k+1]=", x_i-y[i+1:i+k+1], "N_plus_max=", N_plus_max, 's1=', (N_minus_max+N_plus_max)/2)
    return (N_minus_max+N_plus_max)/2
k=3
z=np.zeros_like(y)
for i in range(k, np.size(y)-k):
    z[i]=s1_func_test(k, i, y[i], y)
    # if i>10:
    #     break

fig, ax=plt.subplots(nrows=2, ncols=1, sharex=True)
ax[0].plot(y), ax[0].grid(), ax[0].title.set_text("Raw signal")
ax[1].plot(z), ax[1].grid(), ax[1].title.set_text("s1 values")
plt.show()

#%%
### Compute s1 function for each data points. a holds the value of s1 function at each time index
a=np.zeros(np.size(fnirs_1_data))
k=5
for i in range(k, np.size(fnirs_1_data)-k): # array index is 0, 1, 2, 3 (first i if k=3), 4
    a[i]=help_functions_stress.s1_func(k, i, fnirs_1_data[i], fnirs_1_data) # compute peak function value S1    
plot_raw_fn_diff(fnirs_1_data, a, first_diff, "Raw Data", "S1 function", "First Difference")   
# Use peak detection method 
pks_s1_peaks, _=sc.signal.find_peaks(a, distance=12) # distance=15   rel_height=1e-8
plot_timeseries_marker(idx, fnirs_1_data, pks_s1_peaks, fnirs_1_data[pks_s1_peaks], "Detected peaks: S1")

# Using thresholding method
a=np.zeros(np.size(fnirs_1_data))
k=5
for i in range(k, np.size(fnirs_1_data)-k): # array index is 0, 1, 2, 3 (first i if k=3), 4
    a[i]=help_functions_stress.s1_func(k, i, fnirs_1_data[i], fnirs_1_data) # compute peak function value S1

plot_raw_fn_diff(fnirs_1_data, a, first_diff, "Raw Data", "S1 function", "First Difference")   

# Modification in s1 function based peak detection
threshold_s1, k_c=0.5e-05, math.ceil(0.25/T) # threshold for s1 function # threshold_s1=np.mean(a)+1.5*np.std(a)
max_first_diff=np.max(first_diff)
# get the sample index where objective function (a) is greater than threshold as well as the value of first-difference is
# 0.1 times greater than the max of first-difference values
pks_s1=np.intersect1d(np.where(a>=threshold_s1)[0], np.where(first_diff>=0.1*max_first_diff)[0]) 
plot_timeseries_marker(idx, fnirs_1_data, pks_s1, fnirs_1_data[pks_s1])
#pks_correction=help_functions_stress.remove_false_peak(a, pks_s1, 2) # peaks after removing neighbouring peaks which are close
pks_s1_corr=help_functions_stress.del_false_peak(list(pks_s1), k_c)
# End of Modification in s1 function based peak detection
plot_timeseries_marker(idx, fnirs_1_data, pks_s1_corr, fnirs_1_data[pks_s1_corr], "Detected peaks: S1")
### End of compute S1 function ... #

#%% Generate IBI time series with same size of fnirs_1_data
ibi_ampd=help_functions_stress.ibi_timeseries(pks_ampd, T, len(fnirs_1_data))
ibi_ampd=help_functions_stress.ibi_interpolation(ibi_ampd, method='polynomial', order=2)

ibi_m2d=help_functions_stress.ibi_timeseries(pks_m2d, T, len(fnirs_1_data))
ibi_m2d=help_functions_stress.ibi_interpolation(ibi_m2d, method='polynomial', order=2)

ibi_s1=help_functions_stress.ibi_timeseries(pks_s1_peaks, T, len(fnirs_1_data))
ibi_s1=help_functions_stress.ibi_interpolation(ibi_s1, method='polynomial', order=2)

w=np.array([1/3, 1/3, 1/3])
ibi_series=(w[0]*ibi_ampd + w[1]*ibi_m2d +w[2]*ibi_s1)/np.sum(w) # take average of IBIs
#%%
# # testing the code
# import help_functions_stress
# i=7;
# x_i=fnirs_1_data[i]; 
# print(max(x_i-fnirs_1_data[i-k:i]))
# N_minus_max=max(x_i-fnirs_1_data[i-k:i])
# N_plus_max=max(x_i-fnirs_1_data[i+1:i+k+1])
# for j in np.arange(i-k, i):
#     print(j)
#     print(fnirs_1_data[i]-fnirs_1_data[j])
# q25, q75 = np.percentile(a, [25, 75])
# bin_width = 2 * (q75 - q25) * len(a) ** (-1/3)
# bins = round((a.max() - a.min()) / bin_width)
# plt.hist(a, density= True, bins=30)
# plt.ylabel('Probability')
# plt.ylabel("Data")
# # end of testing the code
#%%
## S5 function for peak detection #
mean_a_pos, std_a_pos=np.mean(a[a>0]), np.std(a[a>0]) # Get mean and std of all positive peak value of peak function
h_s5=1
k_s5=2
peaks_s5=[]
for i, item in enumerate(a):
    if item>0 and (item-mean_a_pos)> (h_s5*std_a_pos):
        peaks_s5.append([i, fnirs_1_data[i]]) # storing index as well as value
# # Get the index of detected peaks which are very close and need to be eliminated
# del_index=[]
# for j in range(1, len(peaks_s5)):
#     if np.abs(peaks_s5[j][0]-peaks_s5[j-1][0])<=k_s5: # check if adjacent peaks are close or not (abs(i-j)<=k). If close,delete the peak with minimum value
#         del_index.append(help_functions_stress.min_value_index(j, peaks_s5))
peaks_s5_test=peaks_s5
j=0
for i, item in enumerate(list(peaks_s5_test)):
    if np.abs(item[0]-peaks_s5_test[j-1][0])<=k_s5: # check if adjacent peaks are close or not (abs(i-j)<=k). If close,delete the peak with minimum value
        peaks_s5_test.remove(item)
        continue
    j=j+1
    
# peaks_s5=np.array(peaks_s5) # convert list into array
# peaks_s5_corr=np.delete(peaks_s5, del_index, axis=0) # delete the adjacent peaks
# peaks_s5_corr=peaks_s5_corr[:,0]
# peaks_s5_corr=peaks_s5_corr.astype(int)
peaks_s5_corr=np.array(peaks_s5_test)[:, 0].astype(int)
plot_timeseries_marker(idx, fnirs_1_data, peaks_s5_corr, fnirs_1_data[peaks_s5_corr], "Detected peaks: S1")
# # end of S5 function for peak detection #
#%%
p=help_functions_stress.first_difference(fnirs_1_data, i, T)


# ## S5 function for peak detection and correction condition is little different(if fnirs_1_data[i]>fnirs_1_data[j], then remove j as peak) #
# mean_a_pos, std_a_pos=np.mean(a[a>0]), np.std(a[a>0]) # Get mean and std of all positive peak value of peak function
# h_s5=1
# k_s5=3
# peaks_s5=[]
# for i, item in enumerate(a):
#     if item>0 and (item-mean_a_pos)> (h_s5*std_a_pos):
#         peaks_s5.append([i, fnirs_1_data[i]]) # storing index as well as value
# # # Get the index of detected peaks which are very close and need to be eliminated
# # del_index=[]
# # for j in range(1, len(peaks_s5)):
# #     if np.abs(peaks_s5[j][0]-peaks_s5[j-1][0])<=k_s5: # check if adjacent peaks are close or not (abs(i-j)<=k). If close,delete the peak with minimum value
# #         del_index.append(help_functions_stress.min_value_index(j, peaks_s5))
# peaks_s5_test=peaks_s5
# j=0
# for i, item in enumerate(list(peaks_s5_test)):
#     # print('i=', i, 'j=', j, 'item=', item, 'peaks_s5_test[j-1][0]', peaks_s5_test[j-1][0])
#     if np.abs(item[0]-peaks_s5_test[j-1][0])<=k_s5: # check if adjacent peaks are close or not (abs(i-j)<=k). If close,delete the peak with minimum value
#         if fnirs_1_data[item[0]] < fnirs_1_data[peaks_s5_test[j-1][0]]:
#             peaks_s5_test.remove(item)
#         else:
#             peaks_s5_test.remove(peaks_s5_test[j-1])
#         # print('deleted: ', item)
#         # print('\nupdated list:\n', peaks_s5_test[:j+1])
#         continue
#     # if j>=10:
#     #     break
#     j=j+1
# peaks_s5_corr=np.array(peaks_s5_test)[:, 0].astype(int)
# plot_timeseries_marker(idx, fnirs_1_data, peaks_s5_corr, fnirs_1_data[peaks_s5_corr])

#%%
### Compute time difference between two peaks
ipi_ampd=help_functions_stress.compute_peak_dist(pks_ampd, T)
ipi_s1=help_functions_stress.compute_peak_dist(pks_s1_corr, T)
ipi_s5=help_functions_stress.compute_peak_dist(peaks_s5_corr, T)


#%%
def generate_heart_rate_sample(idx, ipi, start=0):
    heart_rate=np.zeros((len(idx), 1))
    i=0
    while(i<len(ipi)):
        heart_rate[start:int(ipi[i, 0])+1, 0]=ipi[i, 1]
        start=int(ipi[i, 0])+1
        i=i+1
    return heart_rate    
hr_s5=generate_heart_rate_sample(idx, ipi_s5)
hr_s1=generate_heart_rate_sample(idx, ipi_s1)
hr_ampd=generate_heart_rate_sample(idx, ipi_ampd)
# computed weighted average of heart rate variation
hr=np.average(np.column_stack((hr_ampd, hr_s5, hr_s1)), axis=1, weights=[5.04, 4.39, 3.78])

#%%
# correction of heart rate 
from scipy.interpolate import CubicSpline
time_window=4 # in sec
N=int(np.round(time_window/T)) #  number of samples in the time_window period
i=47378
while i< len(hr):
    SD=np.std(hr[0:i])
    m=np.mean(hr[i-N:i])
    sd=np.std(hr[i-N:i])
    k=0.8*15/sd
    flag=0
    if not (m - k*SD < hr[i] < m + k*SD):
        print("Inside if:", 'i=', i, 'hr[i]=', hr[i])
        for j in range(i, len(hr)):
            flag=1         
            if np.abs(hr[i]-hr[j])>0.1: # checking if two values are almost similar or not
                break
        print("Outside for loop:", "i=%d, hr[i]=%f, j=%d, hr[j]=%f" %(i, hr[i], j, hr[j]))
        f=CubicSpline(range(i-N, i), hr[i-N:i])
        y_new=f(i)
        hr[i:j]=np.squeeze(y_new)
        #i= j if j!=len(hr) else j+1 
        print("after interpolation \n i=%d, j=%d,  hr[i]=%f, hr[j]=%f" %(i, j, hr[i], hr[j]))
        
    i+=1
    # if i >2500:
    #     break
    if flag==1:
        i=j 
#%%    

from itertools import groupby
lst = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
groups = [(k, sum(1 for _ in g)) for k, g in groupby(lst)]

# i1, w1, w2, w3=100, 5.04, 4.39, 3.78
# print(np.dot(np.array((hr_ampd[i1,0], hr_s5[i1, 0], hr_s1[i1, 0])), np.array((5.04, 4.39, 3.78)))/np.sum(np.array((5.04, 4.39, 3.78))))


def correction_s_func(a, threshold, d, mul_factor):
    max_first_diff=np.max(d) # get max of first difference
    # get the sample index where a is greater than threshold as well as mul_factor times greater than max first-difference
    pks_s=np.intersect1d(np.where(a>=threshold)[0], np.where(d>=mul_factor*max_first_diff)[0]) 
    return pks_s


## Detect peaks using s5 function
k5, pks_s5, d_5=3, [], np.zeros(np.size(fnirs_1_data))
for j in range(k5, np.size(fnirs_1_data)-k5-1):    
    d_5[j]=help_functions_stress.first_difference(fnirs_1_data, j, T) # compute first difference at sample i
    if help_functions_stress.s5_func(k5, j, fnirs_1_data[j], fnirs_1_data):        
        pks_s5.append(j)
pks_s5=np.asarray(pks_s5)
pks_s5=np.where(d_5>=0.1*np.max(d_5))[0]

#pks_correction=help_functions_stress.remove_false_peak(?, pks_s5, 2) # peaks after removing neighbouring peaks which are close

# compute time difference between two peaks
#ipi_=help_functions_stress.compute_peak_dist(pks_correction_pre, T)

# Plot the time series data along with red dots at peak position        
#plt.figure()
fig, ax=plt.subplots()
idx=np.arange(0, len(fnirs_1_data))
l=ax.plot(idx, fnirs_1_data)
ax.plot(pks_s1, a[pks_s1], "ro")


m=np.mean(a[pks_s1])


'''
### Cobine s1 and s5 functions for each data points. a holds the value of s1 function at each time index
d_1=a=np.zeros(np.size(fnirs_1_data))
k=2
for i in range(k, np.size(fnirs_1_data)-k): # array index is 0, 1, 2, 3 (first i if k=3), 4
    a[i]=help_functions_stress.s1_func(k, i, fnirs_1_data[i], fnirs_1_data) # compute peak function value S1
    d_1[i]=help_functions_stress.first_difference(fnirs_1_data, i, T) # compute first difference at sample i
# Here we taking the average of all positive values of s1 function
mean_a_pos=np.mean(a[a>0]); std_a_pos=np.std(a[a>0]) # Get mean and std of all positive peak value of peak function
h=3
peaks_s1_5=[]
for i, item in enumerate(a):
    if item>0 and (item-mean_a_pos)> (h*std_a_pos):
        peaks_s1_5.append([i, fnirs_1_data[i]])

# Modification in s1 function based peak detection
threshold_s1=0.5e-5 # threshold for s1 function
max_first_diff=np.max(d_1)
# get the sample index where objective function (a) is greater than threshold as well as first-difference is
# 0.1 times greater than max first-difference
pks_s1=np.intersect1d(np.where(a>=threshold_s1)[0], np.where(d_1>=0.1*max_first_diff)[0]) 
plot_timeseries_marker(idx, fnirs_1_data, pks_s1, a[pks_s1])
pks_correction=help_functions_stress.remove_false_peak(a, pks_s1, 2)
# End of Modification in s1 function based peak detection
plot_timeseries_marker(idx, fnirs_1_data, pks_correction, a[pks_correction])
### End of compute S1 function ... #
''' 



# Data points that are not present in 
idx_not=np.array(list(set(idx)-set(pks_s1)))
plt.hist(a[idx_not])
plt.show()

plt.hist(a[pks_s1])
plt.show()

plt.figure()
fig, ax=plt.subplots()
idx=np.arange(0, len(fnirs_1_data))
l=ax.plot(idx, fnirs_1_data)
ax.plot(idx_not, a[idx_not], "ro")

