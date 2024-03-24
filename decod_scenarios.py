#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
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
# ~ sys.path.insert(1, '/home/pdhara/anaconda3/envs/mne/lib/python3.7/site-packages/mne/')
# ~ import mne
import mne, mne_nirs
from mne_nirs.statistics import run_GLM, glm_region_of_interest
from mne_nirs.visualisation import plot_glm_topo

import pandas as pd
from scipy.stats import ttest_ind, ttest_1samp

from sklearn import svm, preprocessing, linear_model
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold, permutation_test_score
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold # Feature selector
from sklearn.preprocessing import Normalizer, StandardScaler, MinMaxScaler, PowerTransformer, MaxAbsScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV # For optimization
from sklearn.decomposition import PCA
import seaborn as sns # For plotting data
from statistics import mean

sys.path.insert(1, '/data2/pdhara/decoding_pire/iaoToolboxCustom/')
#sys.path.append('/home/pdhara/work/codes/EMOIO/iaoToolboxCustom')
import helpers, io, proc, nirs_help
import supporting_func
#sys.path.insert(1, '/data2/pdhara/decoding_pire/helperFunc.py')
#import helperFunc
#from iaoToolboxCustom import io, proc #BrainvisionInfo, FilterParams, fNIRSreadParams
from proc import fNIRSreadParams, FilterParams, EpochProps


info=fNIRSreadParams(
	preload=True,
	verbose=None
)

filterParams = FilterParams(
	l_freq=0.01,
	h_freq=0.1,
	method='fir',
	fir_design='firwin',
	l_trans_bandwidth=0.005,
	h_trans_bandwidth=0.02,
	phase='zero-double'
)

# epoProps = EpochProps(
# 	tmin=-2, # start time before event. If nothing is provided, default is -0.2 in seconds
# 	tmax=10, # end time after each event. If nothing is provided, defaults is 0.5 in seconds
# 	baseline=(-2.0, 0),
# 	preload=False,
# 	event_id = {'pos': 15, 'neg': 16, 'neu': 17},
# 	reject=None #dict(hbo=80e-6)
# )

def data_loading(data_path):

	# Reading of fNIRS and behavioral data
	fnirs_data=proc.load_fNIRS_data(data_path, info)
	# ~ sam_data = helpers.read_behave_mat(sam_file_path, readable=True)

	### converting raw intensity to optical intensity
	fnirs_data_od=mne.preprocessing.nirs.optical_density(fnirs_data)

	### Converting from optical density to haemoglobin
	fnirs_data_haemo=mne.preprocessing.nirs.beer_lambert_law(fnirs_data_od)
	
	## Drop some channels
	#print('channels before dropping: \n', fnirs_data_haemo.ch_names)
# 	fnirs_data_haemo.drop_channels(['S9_D15 hbo', 'S9_D15 hbr', 'S13_D14 hbo', 'S13_D14 hbr', 'S18_D16 hbo', 'S18_D16 hbr'])
	#print('channels after dropping: \n', fnirs_data_haemo.ch_names)

	## Removing heart rate from signal
	fnirs_data_filt=proc.filter(fnirs_data_haemo.copy(), filterParams, 'bandpass')
# 	fnirs_data_filt=fnirs_data_haemo.copy()
# 	data=fnirs_data_filt.get_data()
# 	data_filt=savgol_filter(data, window_length=19, polyorder =3, axis=1)
# 	print('data.shape():', data.shape, 'data_filt.shape():', data_filt.shape)
# 	fnirs_data_filt._data[:,:]=data_filt
	
	return fnirs_data_haemo, fnirs_data_filt

def plot_t_topo(t, info, title, v=5):
    fig, ax=plt.subplots(1,1)
    # v=5 #max(np.ceil(np.abs(t)))
    mne.viz.plot_topomap(t, info, colorbar=True, sensors=True, vmin=-v, vmax=v, outlines='head', contours=0)
    ax.set_title(title, fontweight='bold')
    return fig
    

baseFolder='/data2/pdhara/work/data/pire_data/03_sub/'
run_1='2021-08-24_001'
raw_haemo_sc1, fnirs_data_filt=data_loading(os.path.join(baseFolder, run_1))

# data_raw=raw_haemo_sc1.get_data()

from scipy import signal, ndimage
ch1_data, times=raw_haemo_sc1[0,:]
ch1_data=ch1_data.T #reshpae(len(times))
'''
https://www.cc.gatech.edu/classes/AY2015/cs4475_summer/documents/smoothing_1D.py
https://pythonexamples.org/python-opencv-image-filter-convolution-cv2-filter2d/
ch1_grad=signal.convolve1d(ch1_data, ker, mode='same') 
ch1_grad=ndimage.convolve(ch1_data, ker)
'''
ker=np.array([-1, 0, 1])
ker=ker/(np.sum(ker) if np.sum(ker)!=0 else 1)
import cv2
ch1_grad_cv=cv2.filter2D(ch1_data, -1, ker)

fig, ax=plt.subplots(2, sharex=True)
plt.suptitle('Channel S1_D1 hbo and differentiation')
ax[0].plot(times, ch1_data)
ax[1].plot(times, ch1_grad_cv)
plt.show()

# fig=plt.figure()
# plt.plot(times, ch1_data, linestyle='solid')
# plt.suptitle('Channel S1_D1 hbo')
# plt.show()
q75, q25=np.percentile(ch1_grad_cv, [75, 25])
iqr=q75-q25




# Select only hbo data
fnirs_data_filt=fnirs_data_filt.pick_types(fnirs='hbo')
## Drop the unwanted channels and data segments
fnirs_data_filt.crop(0, 2992.25, include_tmax=True) # eliminate the noisy data at the end of data recording
#fnirs_data_filt.info['bads']=['S16_D11 hbo','S12_D16 hbo', 'S17_D12 hbo','S13_D17 hbo', 'S18_D13 hbo','S16_D16 hbo','S18_D17 hbo',
                              # 'S22_D16 hbo','S28_D21 hbo', 'S31_D26 hbo','S32_D27 hbo']

times, avg_data=supporting_func.get_time_avged_data(fnirs_data_filt)

# import supporting_func
# _, t0= supporting_func.read_recroding_time(os.path.join(baseFolder, run_1), preload=False, verbose=None)

## Load the event data from excel file
# file_path='/data2/pdhara/work/data/pire_data/2021-05-07/'
# xls_path=os.path.join(baseFolder, 'Subj3', 'start_end_tracker.csv')
# df_sc1=pd.read_csv(xls_path)


base_inter=[(27.47, 84.57),
			(2875.98, 2988.41)]

# Calculate the average baseline data
baseline_1=supporting_func.get_data(fnirs_data_filt.copy(), 'hbo', base_inter[0]).mean(axis=1)
baseline_2=supporting_func.get_data(fnirs_data_filt.copy(), 'hbo', base_inter[1]).mean(axis=1)
base_avg=np.mean([baseline_1, baseline_2], axis=0)[:, np.newaxis]
del baseline_1, baseline_2
## End of calculate the average baseline data ##


## Grouping of channels
SD_list=[[(2, 1), (3, 1), (2, 5), (6, 4), (6, 5), (7, 5)],
		[(1, 1), (1, 2), (1, 3), (3, 3), (4, 3), (3, 6), (4, 7), (7, 6), (8, 6), (8, 3), (8, 7), (9, 7)],
		[(5, 2), (5, 8), (4, 2), (9, 8), (10, 8), (10, 9)],
		[(6, 10), (11, 10), (11, 5), (11, 11), (11, 15), (7, 11), (16, 11), (16, 20), (16, 15), (15, 10), (15, 15), (15, 19), (21, 15), (20, 19), (21, 19), (21, 20)],
		[(12, 11), (12, 6), (12, 12), (12, 16), (8, 12), (13, 7), (13, 12), (13, 13), (18, 17), (13, 17), (23, 17), (23, 22), (23, 21), (22, 21), (22, 16), (22, 20), (16, 16), (17, 16), (17, 12), (17, 17), (17, 21)],
		[(9, 13), (14, 13), (14, 8), (14, 14), (14, 18), (10, 14), (18, 13), (18, 18), (18, 22), (24, 22), (24, 18), (24, 23), (25, 23), (19, 14), (19, 18), (19, 23)],
        [(20, 24), (26, 24), (26, 19), (26, 25), (26, 30), (27, 20), (27, 25), (31, 30), (21, 25)],
        [(27, 26), (22, 26), (28, 26), (31, 26), (28, 21), (28, 27), (28, 31), (29, 27), (23, 27), (32, 27), (32, 31), (31, 31)],
        [(29, 22), (29, 28), (24, 28), (30, 28), (30, 23), (30, 29), (30, 32), (25, 29), (32, 32)]]
grp_ch_names=nirs_help.create_grouping_optodes(SD_list, oxy_type='hbo') # convert source detector pairs into channel name e.g., S1_D1 hbo
grp_ch_index=[[fnirs_data_filt.info.ch_names.index(ch_name) for ch_name in ch_list] for ch_list in grp_ch_names] # Convert channel name into channel index
# End of grouping channels ###

## Combination of channels 
from itertools import combinations
roi_index=list(combinations(list(np.arange(0, len(SD_list))), 2))

def compute_roi_corr(epo_roi, roi_index, corr_type='Pearson'):
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


# Loading epochs and compute features
df_events=nirs_help.read_excel_sheets(os.path.join(baseFolder, run_1, 'events.xlsx')) # loading events
#df_events.loc[df_events['Condition']=='Scenario_2', 'Predict']=True if else False
df_events['Predict']=df_events['Condition'].apply(lambda x: 1 if x=='Scenario_2' else 0) # creating the labels for scenarios

cols=['Onset', 'Offset']
epo_mean, epo_auc, epo_var, epo_max=[], [], [], []
pear_corr, spear_corr, cov_value=[], [], []
for index, row in df_events.iterrows():
    t=(row[cols[0]], row[cols[1]])
    epo_data=supporting_func.get_data(fnirs_data_filt.copy(), 'hbo', t)-base_avg # subtract baseline from epoch
    epo_mean.append(epo_data.mean(axis=1)[np.newaxis, :]) # row: epochs, col: channels
    epo_auc.append(np.sum(np.abs(epo_data), axis=1)[np.newaxis, :])  # row: epochs, col: channels
    epo_var.append(np.var(epo_data, axis=1)[np.newaxis, :])  # row: epochs, col: channels
    epo_max.append(np.amax(epo_data, axis=1)[np.newaxis, :])  # row: epochs, col: channels
    
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

# ### Check the missing channels ###
# import itertools
# ch_list_cr=list(itertools.chain(*grp_ch_names))
# ch_data=fnirs_data_filt.pick_types(fnirs='hbo').ch_names
# ch_missed=supporting_func.get_mismatched_ele(ch_data, ch_list_cr)
# ### End of Check the missing channels ####

### Create dictionary of features #####
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


# Balance the data frame based on number of occurrences of two classes in the dataframe
df_balanced=supporting_func.get_balanced_data(df_epo, scaling_type='upsample')

y=df_balanced.Predict
X=df_balanced.drop('Predict', axis=1)




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
        

# Split the data into training and testing 
#X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=1/3, random_state=0)

#### Create pipeline for training
pipe=Pipeline([
    ('scaler', StandardScaler()),
    ('selector', VarianceThreshold()),
    ('pca', PCA()),
    ('classifier', RidgeClassifier())])
# pipe.fit(X_train, y_train)
# print('Training set score' + str(pipe.score(X_train, y_train)))
# print('Test set score'+ str(pipe.score(X_test, y_test)))
#### End of Create pipeline for training

###### Combination of various parameters and retrain the model
parameters={'scaler': [StandardScaler(), MinMaxScaler(), Normalizer(), MaxAbsScaler()],
            'selector__threshold': [0, 0.001, 0.0015], 
            'pca__n_components': list(np.arange(2, 15, 2)),
            'classifier__alpha': list(np.arange(0.1, 10, 0.1)),
            'classifier__solver': ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag']}

# Split the data into training and testing. Further split the training data 
# into many folds to check the pipeline
results, grids, training_acc, testing_acc=[], [], [], []
kf=KFold(n_splits=3, shuffle=True, random_state=10)
for train_index, test_index in kf.split(X):
    #print('train_index: ', train_index, 'test_index: ', test_index)
    X_train, X_test=X.iloc[train_index], X.iloc[test_index]
    y_train, y_test=y.iloc[train_index], y.iloc[test_index]
    grid_f, train_acc, test_acc=cross_validate_(X_train, y_train, pipe, parameters)       
    final_accu=grid_f.score(X_test, y_test)
    results.append([grid_f, mean(train_acc), mean(test_acc), final_accu])
    
    # grids.append(grid_f)
    # training_acc.append(train_acc)
    # testing_acc.append(test_acc)



# Define training parameters
# grid=GridSearchCV(pipe, parameters).fit(X_train, y_train)
# print('Training set score:' + str(grid.score(X_train, y_train)))
# print('Test set score:'+ str(grid.score(X_test, y_test)))
# best_params=grid.best_params_
# print(best_params)
# # Stores the optimum model in best_pipe
# best_pipe = grid.best_estimator_
# print(best_pipe)
# result_df = pd.DataFrame.from_dict(grid.cv_results_, orient='columns')
# print(result_df.columns)
    
# print('Training Accuracy: {}'.format(np.mean(train_acc)))
# print('Testing Accuracy: {}'.format(np.mean(test_acc)))
###### End of combination of various parameters and retrain the model


##### Create pipeline for Random Forest
pipe_rf=Pipeline([('scl', StandardScaler()),
                  ('clf', RandomForestClassifier(random_state=42))])
params_rf={'scl':[StandardScaler(), MinMaxScaler(), Normalizer(), MaxAbsScaler()],
           'clf__n_estimators': list(np.arange(1, 200, 2))}
grid_rf=GridSearchCV(pipe_rf, params_rf).fit(X_train, y_train)
#pipe_rf.fit(X_train, y_train)
print('Training set score:'+ str(grid_rf.score(X_train, y_train)))
print('Test set score:' + str(grid_rf.score(X_test, y_test)))
##### End of Create pipeline for Random Forest

###### Create pipeline for Decision tree
from sklearn import tree
pipe_dt=Pipeline([('scl', StandardScaler()),
                  ('pca', PCA()),
                  ('clf', tree.DecisionTreeClassifier()) ])
params_dt={'scl': [StandardScaler(), MinMaxScaler(), Normalizer(), MaxAbsScaler()],
           'pca__n_components': list(np.arange(10, 50, 10)),
           'clf__min_samples_split': list(np.arange(1, 8, 1)),
           'clf__max_features':['auto', 'sqrt', 'log2'],
           'clf__random_state': [20, 48]}

grid_dt=GridSearchCV(pipe_dt, params_dt).fit(X_train, y_train)
print('Training set score: '+str(grid_dt.score(X_train, y_train)))
print('Test set score: '+ str(grid_dt.score(X_test, y_test)))
print('Best parameters:'+ str(grid_dt.best_params_))
##### End of Create pipeline for Decision Tree


sns.relplot(data=result_df,
	kind='line',
	x='param_classifier__alpha',
	y='mean_test_score',
	hue='param_scaler',
	col='param_selector__threshold')
plt.show()



# Define training parameters
k_fold=KFold(n_splits=10, shuffle=True, random_state=10)
# create model for SVM classifier
clf=RidgeClassifier(class_weight='balanced').fit(X, y)

clf.score(X, y)

### Combining multiple estimators, along with their corresponding pipeline and hyperparameter tuning. 
#https://stackoverflow.com/questions/63794895/sklearn-pipeline-with-multiple-estimators
pipe_comb=Pipeline([
    ('scl', StandardScaler()),
    ('pca', PCA()),
    ('clf', RidgeClassifier())])
para_comb=[
    {
     'clf': (RidgeClassifier(),),
     'clf__alpha': (0.001, 0.01, 0.1, 1, 10, 100),
     'pca__n_components': list(np.arange(10, 50, 10)),
     },
    {
     'clf': (RandomForestClassifier(),),     
     'clf__n_estimators': (10, 30),
     'pca__n_components': list(np.arange(10, 50, 10)),
     }
    ]
grid_comb=GridSearchCV(pipe_comb, para_comb).fit(X_train, y_train)
print('Training set score:' + str(grid_comb.score(X_train, y_train)))
print('Test set score:'+ str(grid_comb.score(X_test, y_test)))
print('Best parameters:'+ str(grid_comb.best_params_))
##### End of Combinging multiple estimators ... ###



# time_avg_epo_pos=[nirs_help.get_time_avg_epo(epo_data) for epo_data in subs_data_pos]
# auc_epo_pos=[nirs_help.get_area_under_curve(epo_data) for epo_data in subs_data_pos]
# var_epo_pos=[nirs_help.get_variance(epo_data) for epo_data in subs_data_pos]
# max_epo_pos=[nirs_help.get_maximum(epo_data) for epo_data in subs_data_pos]

# log_data_path=os.path.join(baseFolder, 'Subj3', 'start_end_tracker.csv')
# hdr_path=os.path.join(baseFolder, run_1)
# df_events=supporting_func.create_evt_from_log(log_data_path, hdr_path)



