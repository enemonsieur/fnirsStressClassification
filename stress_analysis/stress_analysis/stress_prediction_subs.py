"""
The stress_fnirs.py script read fNIRS data from a path specified in path_dict. The fNIRS data are filtered
in frequency range of heart rate signal. Thereafter, we implemented several methods such as AMPD, M2D, and S1 function
to detect peaks in the time series data. The detected peaks are corresponding to the instance of heart beating.
We computed the interbeat intervals (IBI) two two heart beatings. We computed average IBI in an epoch. 

Also, We have computeed average IBI value for beginning, mid, and end of an epoch. We done t-statistic on the 
computed IBI values between two epxerimental conditions and as well as beginning and end of epochs. 
The results (IBI_values in excel file and t-statistics in text file) are stored in output folder. 

We have done this analysis for all subjects.

Created on Mon 22 16:31:46 2022

@author: pdhara
"""
import os
#from pathlib import Path
import numpy as np
import scipy as sc
import pandas as pd
import matplotlib.pyplot as plt
import sys
import mne, mne_nirs

#sys.path.insert(1, '/data2/pdhara/decoding_pire/iaoToolboxCustom/')
import proc  
#import nirs_help #supporting_func, #helpers, io, proc
import ampd 
from proc import fNIRSreadParams, FilterParams #, EpochProps
import help_functions_stress    
2   
info=fNIRSreadParams(
	preload=True,
	verbose=None)

filterParams = FilterParams(
	l_freq=1,
	h_freq=1.9,
	method='fir',
	fir_design='firwin',
	phase='zero-double')



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
grp_ch_names_hbo=help_functions_stress.flatten_list(help_functions_stress.create_grouping_optodes(SD_list, oxy_type='hbo')) # convert source detector pairs into channel name e.g., S1_D1 hbo
# End of Grouping of channels ...
#%%
baseFolder='/data2/pdhara/work/data/pire_data'

for sub, value in path_dict.items():
    # Reading of fNIRS and behavioral data
    data_path=os.path.join(baseFolder, sub, value[0])
    fnirs_data=proc.load_fNIRS_data(data_path, info)
    ### converting raw intensity to optical intensity
    fnirs_data_od=mne.preprocessing.nirs.optical_density(fnirs_data)
    ### Converting from optical density to haemoglobin
    fnirs_data_haemo=mne.preprocessing.nirs.beer_lambert_law(fnirs_data_od)
    raw_resampled = fnirs_data_haemo.resample(10) # we resampled the data to match methods mentioned in the paper
    ## Removing heart rate from signal
    fnirs_data_filt=proc.filter(raw_resampled.copy(), filterParams, 'bandpass')
    # Sampling interval
    T=1/fnirs_data_filt.info['sfreq'] # sampling period
    
    #%%
    ibi_series_ch=[] #np.zeros((len(grp_ch_names_hbo), len(fnirs_data_filt.times)))
    # Iterate over channels to extract interbeat intervals for each channel
    for ch_name in grp_ch_names_hbo:
        fnirs_1=fnirs_data_filt.copy().pick_channels([ch_name]).pick_types(fnirs="hbo")
        fnirs_1_ch=fnirs_1.get_data().flatten()
        idx=np.arange(0, len(fnirs_1_ch)) # needed for plotting peaks in time series
        #%%
        ### Detect peaks using AMPD algorithm. pks holds the peak index
        debug=True
        if debug:
            pks_ampd, ZLSM, l_scale=ampd.find_peaks_original(fnirs_1_ch, debug=True)
        else:
            pks_ampd=ampd.find_peaks_original(fnirs_1_ch)
        
        # Plot the time series data along with red dots at peak position
        
        #plot_timeseries_marker(idx, fnirs_1_ch, pks_ampd, fnirs_1_ch[pks_ampd], "Detected peaks: AMPD")
        # End of Detect Peaks using AMPD ...
        #%% Compute first difference of time series and its first difference
        first_diff=help_functions_stress.first_difference_series(fnirs_1_ch, T)
        max_first_diff=np.max(first_diff)
    
        #%%
        # Detect peaks using M2D algorithm
        start=3
        m2d_values=np.zeros(np.size(fnirs_1_ch))
        for i in range(start, np.size(fnirs_1_ch)-start):
            m2d_values[i]=help_functions_stress.m2d(fnirs_1_ch, i)
      
        # Use peak detection method 
        pks_m2d, _=sc.signal.find_peaks(m2d_values, distance=12) # distance=15   rel_height=1e-8    
        
        #plot_timeseries_marker(idx, fnirs_1_ch, pks_m2d, fnirs_1_ch[pks_m2d], "Detected peaks: M2D")
        # End of Detect peaks using M2D algorithm
        
        #%%
        ### Compute s1 function for each data points. a holds the value of s1 function at each time index
        a=np.zeros(np.size(fnirs_1_ch))
        k=5
        for i in range(k, np.size(fnirs_1_ch)-k): # array index is 0, 1, 2, 3 (first i if k=3), 4
            a[i]=help_functions_stress.s1_func(k, i, fnirs_1_ch[i], fnirs_1_ch) # compute peak function value S1    
        #plot_raw_fn_diff(fnirs_1_ch, a, first_diff, "Raw Data", "S1 function", "First Difference")   
        # Use peak detection method 
        pks_s1_peaks, _=sc.signal.find_peaks(a, distance=12) # distance=15   rel_height=1e-8
        #plot_timeseries_marker(idx, fnirs_1_ch, pks_s1_peaks, fnirs_1_ch[pks_s1_peaks], "Detected peaks: S1")
    
    
        #%% Generate IBI time series with same size of fnirs_1_data
        ibi_ampd=help_functions_stress.ibi_timeseries(pks_ampd, T, len(fnirs_1_ch))
        ibi_ampd=help_functions_stress.ibi_interpolation(ibi_ampd, method='polynomial', order=2)
        
        ibi_m2d=help_functions_stress.ibi_timeseries(pks_m2d, T, len(fnirs_1_ch))
        ibi_m2d=help_functions_stress.ibi_interpolation(ibi_m2d, method='polynomial', order=2)
        
        ibi_s1=help_functions_stress.ibi_timeseries(pks_s1_peaks, T, len(fnirs_1_ch))
        ibi_s1=help_functions_stress.ibi_interpolation(ibi_s1, method='polynomial', order=2)
        
        w=np.array([1/3, 1/3, 1/3])
        ibi_1_ch=(w[0]*ibi_ampd + w[1]*ibi_m2d +w[2]*ibi_s1)/np.sum(w) # take average of IBIs
        # store all the computed IBIs in a list
        ibi_series_ch.append(ibi_1_ch)
    ibi_series=np.vstack(ibi_series_ch).mean(axis=0) # take average across channels
        
    #%%
    # Loading epochs for computing average interbeat intervals (IBIs) in a epoch
    df_events=help_functions_stress.read_excel_sheets(os.path.join(data_path, 'events.xlsx')) # loading events
    #df_events.loc[df_events['Condition']=='Scenario_2', 'Predict']=True if else False
    df_events['Predict']=df_events['Condition'].apply(lambda x: 1 if x=='Scenario_2' else 0) # creating the labels for scenarios
    
    N_ibi=30 # Compute average IBI within these many samples at the beginning and end of an epoch 
    ibi_list=[]
    for index, row in df_events.iterrows():
        # added 8 sec on onset time as fNIRS signal has a delay of around 5-8 sec
        idxs_evt=fnirs_data_filt.time_as_index([row["Onset"]+8, (row["Onset"]+row['Offset'])/2, row['Offset']]) # beginning, mid, and end of an epoch
        # compute average IBI within a whole epoch, begining of epoch, end of epoch
        ibi_list.append([np.mean(ibi_series[idxs_evt[0]:idxs_evt[-1]]), np.mean(ibi_series[idxs_evt[0]: idxs_evt[0]+N_ibi]),
                       np.mean(ibi_series[idxs_evt[1]: idxs_evt[1]+N_ibi]),  np.mean(ibi_series[idxs_evt[-1]-N_ibi: idxs_evt[-1]])])
        
    df_ibi=pd.DataFrame(ibi_list, columns=["ibi_mean", "ibi_begin", "ibi_mid", "ibi_end"]) # create a dataframe with average IBIs
    df_ibi=pd.concat([df_events, df_ibi], axis=1) # combine two dataframes 
    # Store in a excel file in the path 'output'
    df_ibi.to_excel(os.path.join("output", sub+"__df_avg_ibi.xlsx"))
    
    #%% t-test between scenarios and experimental conditions
    # two sample t-test of average IBIs between two scenarios (power outage and underload conditions)
    t_sc_0_1, p_sc_0_1=sc.stats.ttest_ind(df_ibi[df_ibi['Predict']==0]['ibi_mean'].values, 
                                           df_ibi[df_ibi['Predict']==1]['ibi_mean'].values)
    
    # pairwise t-test of IBI rate between beginning and end of an epoch
    t_begin_end, p_begin_end=sc.stats.ttest_rel(df_ibi['ibi_begin'], df_ibi['ibi_end'], alternative='two-sided')
    
    # write in a text file in the path "output"
    text=["Between Experimental Conditions \n", "t_value_sc_0_vs_1="+str(t_sc_0_1)+ "\n", "p_value_sc_0_vs_1="+str(p_sc_0_1) + "\n\n",
          "Between beginning and End of events \n", "t_value_begin_end="+str(t_begin_end) + "\n", "p_value_begin_end="+str(p_begin_end) + "\n\n",]
    with open(os.path.join('output', sub+'_t_stats.txt'), 'w') as f:
        f.writelines(text)