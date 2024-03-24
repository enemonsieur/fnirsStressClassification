#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This scrip is written to decode two experimental conditions: power outage condition, 
underload condition. Here we computed several features in ROIs. Computed features are 
feed into ML models for predicting two conditions.

The results are shown in bar plot


Created on Tue Aug 24 15:49:48 2021

@author: pdhara
"""
import os, pathlib
import numpy as np
import matplotlib.pyplot as plt
from itertools import compress
from pathlib import Path
import glob as glob
import sys
import mne, mne_nirs
from mne_nirs.statistics import run_GLM, glm_region_of_interest
from mne_nirs.visualisation import _plot_GLM_topo

import pandas as pd
from scipy.stats import ttest_ind, ttest_1samp
from itertools import combinations


from sklearn import svm, preprocessing, linear_model
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold, permutation_test_score
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold # Feature selector
from sklearn.preprocessing import Normalizer, StandardScaler, MinMaxScaler, PowerTransformer, MaxAbsScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV # For optimization
from sklearn.decomposition import PCA
import seaborn as sns # For plotting data
from statistics import mean

sys.path.insert(1, 'C:/Users/ene Monsieur/OneDrive/Documents/OFFIS.de/fNIRS_data_analysis/Prasenjit_Datas/stress_analysis/decoding_pire/iaoToolboxCustom')
import proc, nirs_help
import supporting_func
from proc import fNIRSreadParams, FilterParams


info=fNIRSreadParams(
	preload=True,
	verbose=None
)
filterParams = FilterParams(
	l_freq=0.01,
	h_freq=0.1,
    	method='fir',
    	fir_design='firwin',
    	l_trans_bandwidth=0.01,
    	h_trans_bandwidth=0.01,
    	phase='zero-double'
    )

def data_loading(data_path):

	# Reading of fNIRS and behavioral data
	fnirs_data=proc.load_fNIRS_data(data_path, info)

	### converting raw intensity to optical intensity
	fnirs_data_od=mne.preprocessing.nirs.optical_density(fnirs_data)

	### Converting from optical density to haemoglobin
	fnirs_data_haemo=mne.preprocessing.nirs.beer_lambert_law(fnirs_data_od)
	
	## Removing heart rate from signal
	fnirs_data_filt=proc.filter(fnirs_data_haemo.copy(), filterParams, 'bandpass')
	
	return fnirs_data_haemo, fnirs_data_filt

def plot_t_topo(t, info, title, v=5):
    fig, ax=plt.subplots(1,1)
    mne.viz.plot_topomap(t, info, colorbar=True, sensors=True, vmin=-v, vmax=v, outlines='head', contours=0)
    ax.set_title(title, fontweight='bold')
    return fig

def compute_roi_corr(epo_roi, roi_index, corr_type='Pearson'):
    """
    Compute correlation between two time series data in two ROIs. 

    Parameters
    ----------
    epo_roi : TYPE
        DESCRIPTION.
    roi_index : TYPE
        DESCRIPTION.
    corr_type : string, Pearson or Spearman correlation
        DESCRIPTION. The default is 'Pearson'.

    Raises
    ------
    Exception
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    from scipy import stats
    corr_type_list=['Pearson', 'Spearman']
    if corr_type not in corr_type_list:
        raise Exception('Invalid correlation type')
        return
    corr_val=[]
    for tup in roi_index:
        if corr_type=='Pearson':
            r,_,_,_,_=nirs_help.pearsonr_ci(epo_roi[:, tup[0]], epo_roi[:, tup[1]])
        elif corr_type=='Spearman':
            r,_=stats.spearmanr(epo_roi[:, tup[0]], epo_roi[:, tup[1]])
        corr_val.append(r)
    return np.asarray(corr_val)

def cross_validate_(X_train, y_train, pipeline, para):
    train_acc=[]
    test_acc=[]
    cv=KFold(n_splits=5, shuffle=True, random_state=10)
    grid=GridSearchCV(pipeline, para)
    for train_ind, vali_ind in cv.split(X_train):
        X_t, y_t=X_train.iloc[train_ind], y_train.iloc[train_ind]
        train_score=grid.fit(X_t, y_t).score(X_t, y_t)
        train_acc.append(train_score)
        
        X_vali, y_vali=X_train.iloc[vali_ind], y_train.iloc[vali_ind]
        test_score=grid.score(X_vali, y_vali)
        test_acc.append(test_score)
        
    return grid, train_acc, test_acc


#%% Grouping of channels for ROIs (here 9 ROIs:)
# Anterior (left, midline, right), central (left, midline, right), posterior (left, midline, right)
SD_list=[[(2, 1), (3, 1), (2, 5), (6, 4), (6, 5), (7, 5)],
		[(1, 1), (1, 2), (1, 3), (3, 3), (4, 3), (3, 6), (4, 7), (7, 6), (8, 6), (8, 3), (8, 7), (9, 7)],
		[(5, 2), (5, 8), (4, 2), (9, 8), (10, 8), (10, 9)],
		[(6, 10), (11, 10), (11, 5), (11, 11), (11, 15), (7, 11), (16, 11), (16, 20), (16, 15), (15, 10), (15, 15), (15, 19), (21, 15), (20, 19), (21, 19), (21, 20)],
		[(12, 11), (12, 6), (12, 12), (12, 16), (8, 12), (13, 7), (13, 12), (13, 13), (18, 17), (13, 17), (23, 17), (23, 22), (23, 21), (22, 21), (22, 16), (22, 20), (16, 16), (17, 16), (17, 12), (17, 17), (17, 21)],
		[(9, 13), (14, 13), (14, 8), (14, 14), (14, 18), (10, 14), (18, 13), (18, 18), (18, 22), (24, 22), (24, 18), (24, 23), (25, 23), (19, 14), (19, 18), (19, 23)],
        [(20, 24), (26, 24), (26, 19), (26, 25), (26, 30), (27, 20), (27, 25), (31, 30), (21, 25)],
        [(27, 26), (22, 26), (28, 26), (31, 26), (28, 21), (28, 27), (28, 31), (29, 27), (23, 27), (32, 27), (32, 31), (31, 31)],
        [(29, 22), (29, 28), (24, 28), (30, 28), (30, 23), (30, 29), (30, 32), (25, 29), (32, 32)]]
grp_ch_names=nirs_help.create_grouping_optodes(SD_list, oxy_type='hbr') # convert source detector pairs into channel name e.g., S1_D1 hbo
#%% End of grouping channels ###    
# key: sub_name; value: folder inside sub_name, log_data folder, baseline intervals (beginning and end of experiment)
path_dict={'06_sub': ['2021-10-06_003', 'log_data', [(27.90, 142.47), (4630.59, 4732.28)]],
			'07_sub': ['2021-10-25_001', 'log_data', [(15.39, 161.10), (3700.05, 3825.94)]],
			#'08_sub': ['2021-10-26_002', 'log_data', [(25.22, 112.73),	(4377.42, 4456.86)]],
			#'09_sub': ['2021-10-28_002', 'log_data', [(52.87, 127.97),	(4072.61, 4145.96)]],
			#'10_sub': ['2021-11-02_003', 'log_data', [(46.80, 120.55),	(4351.66, 4474.44)]],
			#'11_sub': ['2021-11-05_002', 'log_data', [(30.44, 152.23),	(4687.38, 4832.41)]],
			#'12_sub': ['2021-11-08_001', 'log_data', [(102.03, 222.54),	(5005.38, 5130.31)]],
			#'13_sub': ['2021-11-09_001', 'log_data', [(43.77, 191.85),	(4302.88, 4526.49)]],
			#'14_sub': ['2021-11-11_001', 'log_data', [(82.07, 205.04),	(4901.64, 5023.67)]],
            #'15_sub': ['2021-11-18_001', 'log_data', [(63.29, 220.40),	(3851.89, 3983.02)]]
			}
#%% Parameters for ML model training
#### Create pipeline for training ML model
pipe=Pipeline([
    ('scaler', StandardScaler()),
    ('selector', VarianceThreshold()),
    ('pca', PCA()),
    ('classifier', SVC())])
#### End of Create pipeline for training

###### Combination of various parameters and retrain the model
parameters={'scaler': [StandardScaler(), MinMaxScaler(), Normalizer(), MaxAbsScaler()],
            'selector__threshold': [0, 0.001, 0.0015], 
            'pca__n_components': list(np.arange(2, 20, 2)),
            'classifier__alpha': list(np.arange(0.1, 10, .5)),
            # 'classifier__solver': ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag']
            }
# end of Parameters for ML training
#%%
# Loading data
baseFolder='C:/Users/ene Monsieur/OneDrive/Documents/OFFIS.de/fNIRS_data_analysis/Prasenjit_Datas/Data_PIRE_'

results_subs=[]
for sub, value in path_dict.items():
    raw_haemo_sc1, fnirs_data_filt=data_loading(os.path.join(baseFolder, sub, value[0]))
    
    ## Drop the unwanted channels and data segments
    # fnirs_data_filt.crop(0, 2992.25, include_tmax=True) # eliminate the noisy data at the end of data recording
    
    # Select only hbr data
    fnirs_data_filt=fnirs_data_filt.pick_types(fnirs='hbr')
    # Why only hbr and not dhbr? Article
    times, avg_data=supporting_func.get_time_avged_data(fnirs_data_filt)     
    base_inter=value[2] # value [2] holds the baseline interval for each subject 

    # Calculate the average baseline data
    baseline_1=supporting_func.get_data(fnirs_data_filt.copy(), 'hbr', base_inter[0]).mean(axis=1)
    baseline_2=supporting_func.get_data(fnirs_data_filt.copy(), 'hbr', base_inter[1]).mean(axis=1)
    base_avg=np.mean([baseline_1, baseline_2], axis=0)[:, np.newaxis] # take average of baseline data
    del baseline_1, baseline_2
    ## End of calculate the average baseline data ##

    # Grouping of channels based on ROIs
    grp_ch_index=[[fnirs_data_filt.info.ch_names.index(ch_name) for ch_name in ch_list] for ch_list in grp_ch_names] # Convert channel name into channel index

    ## Combination of channels 
    roi_index=list(combinations(list(np.arange(0, len(SD_list))), 2))

    #%% Loading epochs and compute features
    df_events=nirs_help.read_excel_sheets(os.path.join(baseFolder, sub, value[0], 'events.xlsx')) # loading events
    df_events['Predict']=df_events['Condition'].apply(lambda x: 1 if x=='Scenario_2' else 0) # creating the labels for scenarios
    cols=['Onset', 'Offset']
    epo_mean, epo_auc, epo_var, epo_max=[], [], [], []
    pear_corr, spear_corr, cov_value=[], [], []
    for index, row in df_events.iterrows():
        # t=(row[cols[0]], row[cols[1]]) # take the whole epoch
        t=(row[cols[0]]+10, row[cols[0]]+70) # take data after 10 sec of each epoch to up to 70 sec
        epo_data=supporting_func.get_data(fnirs_data_filt.copy(), 'hbr', t)-base_avg # subtract baseline from epoch
        epo_mean.append(epo_data.mean(axis=1)[np.newaxis, :])            # row: epochs, col: channels
        epo_auc.append(np.sum(np.abs(epo_data), axis=1)[np.newaxis, :])  # row: epochs, col: channels
        epo_var.append(np.var(epo_data, axis=1)[np.newaxis, :])          # row: epochs, col: channels
        epo_max.append(np.amax(epo_data, axis=1)[np.newaxis, :])         # row: epochs, col: channels
        
        # average the time series based on ROIs and then compute connectivity measures
        epo_roi=nirs_help.get_row_avg_feat_idx(epo_data.T, grp_ch_index)
        pear_corr.append(compute_roi_corr(epo_roi, roi_index, 'Pearson'))
        spear_corr.append(compute_roi_corr(epo_roi, roi_index, 'Spearman'))

    
    # Create a dictionary that holds the horizontally stacked computed features[row: no of epochs; col: no. of channels]
    epo_feat={'mean':np.vstack(epo_mean), 'auc': np.vstack(epo_auc), 'var': np.vstack(epo_var), 'max': np.vstack(epo_max)}
    # Merge the features based on ROIs (row: epochs; cols: roi)
    epo_feat_roi={} # holds the features for ROI
    for feat_key in epo_feat:
        epo_feat_roi[feat_key+'_roi']=nirs_help.get_row_avg_feat_idx(epo_feat[feat_key], grp_ch_index)
    # append the connectivity features into 'epo_feat_roi' dictionary
    epo_feat_roi.update({'pearson_corr': np.vstack(pear_corr), 'spearman_corr': np.vstack(spear_corr)})

    #%% Create dictionary of features #####
    X_epo=[]
    cols_name=[]
    for key in epo_feat_roi:
        X_epo.append(epo_feat_roi[key]) # create a list of features
        cols_name.append([key+ '_' + str(i+1) for i in np.arange(epo_feat_roi[key].shape[1])]) # genearate features name
    # Horizontally stack all the features (rows: epochs, col: features)
    X_epo=np.hstack(X_epo)
    y=df_events['Predict'].to_numpy()[:,np.newaxis] # get the value as an array that we will to predict
    X_epo=list(np.hstack((X_epo, y))) # stick the features data and prediction data into a list

    # get the name of the columns from dictionary
    from itertools import chain
    cols_name=list(chain.from_iterable(cols_name))
    cols_name.append('Predict')
    # Create dataframe of features
    df_epo=pd.DataFrame(X_epo, columns=cols_name) # create dataframe for the features and prediction value
    ### End of Create dictionary of features ####
    
    
    #%% Balance the data frame based on number of occurrences of two classes in the dataframe
    df_balanced=supporting_func.get_balanced_data(df_epo, scaling_type='upsample') #'downsample', 'upsample'
    
    y=df_balanced.Predict
    X=df_balanced.drop('Predict', axis=1)        

    #%% Split the data into training and testing     
    # Split the data into training and testing. Further split the training data 
    # into many folds to check the pipeline
    results, grids, training_acc, testing_acc=[], [], [], []
    kf=KFold(n_splits=3, shuffle=True, random_state=10)
    for train_index, test_index in kf.split(X):
        #print('train_index: ', train_index, 'test_index: ', test_index)
        X_train, X_test=X.iloc[train_index], X.iloc[test_index]
        y_train, y_test=y.iloc[train_index], y.iloc[test_index]
        # get the best grid model, training accuracy, and testing accuracy
        grid_f, train_acc, test_acc=cross_validate_(X_train, y_train, pipe, parameters)       
        # Now test the model using actual testing data and get actual accuracy
        final_accu=grid_f.score(X_test, y_test)
        # Store the grid, mean_training_accuracy_in_cross_validation, mean_testing_accuracy_in_cross_validation, 
        # and testing_accuracy. For 3 splits of data, get 3 different list of values for each subject
        results.append([grid_f, mean(train_acc), mean(test_acc), final_accu]) 
    
    results_subs.append(results)

# Get the result data as an array for plotting
res_arr_final=[]
for lst in results_subs:
     arr_lst=[]
    # Take result of each subject and results of k (3) folds data
    for el in lst:
        #for each subject, make a list of mean_training_accuracy_in_cross_validation, mean_testing_accuracy_in_cross_validation, 
        # testing_accuracy for k (3) folds 
        arr_lst.append(el[1:]) # except grid take all the result columns--mean_
    arr_lst=np.vstack(arr_lst) # create an array of results from k (3) folds
    # create an array to store mean and std of accuracies (training_validation, testing_validation, final score)
    arr=np.zeros(np.mean(arr_lst, axis=0).shape[0]+np.std(arr_lst, axis=0).shape[0])
    arr[::2]=np.mean(arr_lst, axis=0) # store mean value at every even column indexes (0, 2, 4)
    arr[1::2]=np.std(arr_lst, axis=0) # store std value at every odd column indexes (1, 3, 5)
    res_arr_final.append(arr)
res_arr_final=np.vstack(res_arr_final) # column 4 (final accuracy) and 5 (std of final accuracy)

#%% plotting the results
def bar_dual_err_plot(bar1, bar2, yer1, yer2, barWidth):
    """
    Bar plots of upsampling and downsampling the unbalanced data set

    Parameters
    ----------
    bar1 : array of prediction accuracy : holds accuracy value from downsampling scheme
        DESCRIPTION.
    bar2 : array of prediction accuracy : holds accuracy value from upsampling scheme
        DESCRIPTION.
    yer1 : TYPE
        DESCRIPTION.
    yer2 : TYPE
        DESCRIPTION.
    barWidth : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    r1=np.arange(len(bar1))
    r2=[x+barWidth for x in r1]
    
    # create blue bars
    plt.bar(r1, bar1, width=barWidth, color='b', edgecolor='black', yerr=yer1, capsize=7, label='Downsample')
    # Create cyan bars
    plt.bar(r2, bar2, width=barWidth, color='g', edgecolor='black', yerr=yer2, capsize=7, label='Upsample')
    # General layout
    plt.xticks([r+barWidth for r in range(len(bar1))], ['sub'+str(i+1) for i in range(len(bar1))], fontsize=18)
    plt.ylabel('Accuracy', fontsize=18)
    plt.legend()
    plt.show()

# Store the result for downsampling 
res_arr_final_downsample=res_arr_final # select whether updample or downsample (line 263)
bar1, yer1=res_arr_final_downsample[:, 4], res_arr_final_downsample[:, 5]
# run the above code one more time with parameter 'upsampling' for balancing dataset
res_arr_final_upsample=res_arr_final
bar2, yer2=res_arr_final_upsample[:, 4], res_arr_final_upsample[:, 5]
bar_dual_err_plot(bar1, bar2, yer1, yer2, barWidth=0.3)

#%%
import pickle
info_file = info
with open('info_file.pkl', 'wb') as file:
	
	# A new file will be created
	pickle.dump(info_file, file)




