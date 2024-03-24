# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23   12:59:15 2023

@author: ene Monsieur
"""

"""

    Created on Mon 22 16:31:46 2022

@author: Njeukwa Bounkeu, Ntsanyem
"""
import os
#from pathlib import Path
import numpy as np
import scipy as sc
import pandas as pd
import matplotlib.pyplot as plt
import sys
import mne, mne_nirs
#append the direction of all the toolboxes
sys.path.append('C:/Users/ene Monsieur/OneDrive - Carl von Ossietzky Universität Oldenburg/Documents/OFFIS.de/fNIRS_data_analysis/Prasenjit_Datas/stress_analysis')
sys.path.append('C:/Users/ene Monsieur/OneDrive - Carl von Ossietzky Universität Oldenburg/Documents/OFFIS.de/fNIRS_data_analysis/Prasenjit_Datas/stress_analysis/decoding_pire/')

sys.path.insert(1, 'C:/Users/ene Monsieur/OneDrive - Carl von Ossietzky Universität Oldenburg/Documents/OFFIS.de/fNIRS_data_analysis/Prasenjit_Datas/stress_analysis/decoding_pire/iaoToolboxCustom')
import proc_pire
import nirs_help
import supporting_func, helpers, io, proc_pire
import ampd 
from proc_pire import fNIRSreadParams, FilterParams #, EpochProps
import help_functions_stress    
2   
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold, permutation_test_score
from sklearn import svm, preprocessing, linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.utils import resample


#Info is necessary for extracting hbr signal
info=fNIRSreadParams(
	preload=True,
	verbose=None)

# filter parameters to extract HRV
filterParams = FilterParams(
	l_freq=1,
	h_freq=1.9,
	method='fir',
	fir_design='firwin',
	phase='zero-double')

# filter parameters to extract PFC activity
filterParams_all = FilterParams(
	l_freq=0.01,
	h_freq=0.7,
	method='fir',
	fir_design='firwin',
	phase='zero-double')

def t_test(X,Y):
    p_values = []
    for i in range(X_train.shape[1]):
        feature = X_train[:,i]
        t_test, p_value = stats.ttest_ind(feature[Y==0], feature[Y==1])
        p_values.append(p_value)
    return p_values

    #%% Create the path to the entire folder
path_dict={      '06_sub': ['2021-10-06_003', 'log_data', [(27.90, 142.47), (4630.59, 4732.28)]],
    			'07_sub': ['2021-10-25_001', 'log_data', [(15.39, 161.10), (3700.05, 3825.94)]],
    			'08_sub': ['2021-10-26_002', 'log_data', [(25.22, 112.73),	(4377.42, 4456.86)]],
    			'10_sub': ['2021-11-02_003', 'log_data', [(46.80, 120.55),	(4351.66, 4474.44)]],
    			'11_sub': ['2021-11-05_002', 'log_data', [(30.44, 152.23),	(4687.38, 4832.41)]],
    			'12_sub': ['2021-11-08_001', 'log_data', [(102.03, 222.54),	(5005.38, 5130.31)]],
    			'13_sub': ['2021-11-09_001', 'log_data', [(43.77, 191.85),	(4302.88, 4526.49)]],
    			'14_sub': ['2021-11-11_001', 'log_data', [(82.07, 205.04),	(4901.64, 5023.67)]],
                '15_sub': ['2021-11-18_001', 'log_data', [(63.29, 220.40),	(3851.89, 3983.02)]],
                '16_sub': ['2021-11-19_001', 'log_data', [(254.62, 351.98),	(4602.82, 4612.69)]],
                '17_sub': ['2021-11-22_001', 'log_data', [(212,	284.69),	(3901.64,3950.72)]]
                #'18_sub': ['2021-11-23_001', 'log_data', [(63.29, 220.40),	(3851.89, 3983.02)]],
                #'19_sub': ['2021-12-13_003', 'log_data', [(63.29, 220.40),	(3851.89, 3983.02)]]
    			}


## Grouping of channels for stress prediction:  frontal region/frontal channels for heart rate
SD_list=[[(2, 1), (3, 1), (2, 5), (6, 5), (7, 5)],
    		[(1, 1), (1, 2), (1, 3), (3, 3), (4, 3), (3, 6), (4, 7), (7, 6), (8, 6), (8, 3), (8, 7), (9, 7)],
    		[(5, 2), (5, 8), (4, 2), (9, 8), (10, 8), (10, 9)]]
grp_ch_names_hbo=help_functions_stress.flatten_list(help_functions_stress.create_grouping_optodes(SD_list, oxy_type='hbr')) # convert source detector pairs into channel name e.g., S1_D1 hbo
#%% FILTERING
baseFolder='C:/Users/ene Monsieur/OneDrive - Carl von Ossietzky Universität Oldenburg/Documents/OFFIS.de/fNIRS_data_analysis/Prasenjit_Datas/Data_PIRE_/' #load the base folder of all the subjects

results_subs=[] #init the params for accuracies, ROC...
acc = []
acc_subs = np.zeros(len(path_dict))
acc_boot_subs = np.zeros(len(path_dict))
error_boot_subs = np.zeros((2,len(path_dict)))
n_subjects = len(path_dict)
for sub, value in path_dict.items():  #loop over all subjects
    #take the indices of each subj (will serve later for the plot)
    sub_idx = list(path_dict.keys()).index(sub)
    results_subs =[]
    # Reading of fNIRS and behavioral data
    data_path=os.path.join(baseFolder, sub, value[0])
    raw_intensity=mne.io.read_raw_nirx(data_path, verbose=True, preload=True)
    sample_rate = raw_intensity.info['sfreq']


    # Convert to optical density
    raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)
    #convert to haemo
    raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=0.1)
    
    #baseline correction
    # Set the baseline period
    baseline = (None, 0) # means from the start of time to time zero
    
    # Apply baseline correction
    raw_haemo.apply_function(lambda x: x - x.mean(), picks=mne.pick_types(raw_haemo.info, fnirs='hbr'), verbose=True)
    # Filtering the HRV and artifacts out 
    
    #first resample the data to match the fs mentionned in the paper
    raw_resampled = raw_haemo.resample(10)  
    raw_haemo_hrv=proc_pire.filter(raw_resampled.copy(), filterParams, 'bandpass')
    
    # filter in only the HRV
    raw_haemo = raw_haemo.copy().filter(0.05, 0.7, h_trans_bandwidth=0.2,
                                 l_trans_bandwidth=0.02)  
    
    
     # Load the events excel file to pick the epochs from the data 
    df_events=nirs_help.read_excel_sheets(os.path.join(baseFolder, sub, value[0], 'events.xlsx')) # loading events
    df_events['Predict_help']=df_events['Condition'].apply(lambda x: 1 if x=='Scenario_2' else 0) # creating the labels for scenarios
    cols=['Onset', 'Offset']

      
            #%% Extract the features
    X = [] # initialize feature matrix X
    Y = []
    X_ibi = [] #init matrix for only the IBIS
    nFeat = 2  #define how many parts of the datas you want ex: to find the mean of 2 parts of the data
    cols=['Onset', 'Offset'] #define the cols to extract epochs from events.xlsx
    
    # FEATURE EXTRACTION
    for index, row in df_events.iterrows(): #iterate through each epochs
    
        # Extract the time-series*channels of each epoch, and keep the channels of the PFC only
        epo_data = []  #df of the channels*timeseries
        ch_data = [] #df of each channel
        t=(row[cols[0]]+10 , row[cols[0]]+30)                                        #define the range of time to extract fNIRS data
        for ch_name in grp_ch_names_hbo:                                            # loop through the ROI channels (the PFC) - This will discard unuseful channels
            ch_data_hbr = raw_haemo.copy().pick_channels([ch_name]).pick_types(fnirs='hbr') #extract the hbr only
            ch_data =ch_data_hbr.crop(tmin=t[0], tmax=t[1], include_tmax=True).get_data()   #extract the hbr only
            if len(epo_data) == 0: #init the length of the matrix so that the others ch_data can fit*
                epo_data = ch_data
            else:
                epo_data = np.concatenate((epo_data, ch_data), axis=0)        #stack the choosen channels in epo_data
                
        # Init the length of the time serie and channels of the epo_data
        len_time =len(epo_data[0])                                            #length of the timeseries that will take all the mean
        len_channels= len(epo_data) 
        window_size = len_time // nFeat                                        #define the size of the block of timeSeries we will extract features from
        len_feat = len_channels*nFeat + len_channels*nFeat                     #lenght of the  channels
        
        #init the dataframe of the features we will extract: Mean, Slope, SD, Kurtosis...
        mean_array = np.zeros((len_time // window_size,len_channels)).T    #init the array that takes all the means after the windowshift on all channels
        #slopes = []                                                        #init the slope
        sd_array = np.zeros((len_time // window_size,len_channels)).T      #init the array that takes all the means after the windowshift on all channels
        kurt_array = np.zeros((len_time // window_size,len_channels)).T    #init the array that takes all the means after the windowshift on all channels
        
        
       #loop through parts of the timeseries to extract data from them
        for i in range(len_time // window_size): #define the portions of the time series to extract
            start = i * window_size #define start and end of the block of time series
            end = (i+1) * window_size
            epo_window = epo_data[:,start:end] #extract that part
            mean_array[:,i] = np.mean(epo_window,axis=1) # Calculate the mean of that block 
            
            #find the mean slopes
            #first = epo_window[:,0] #define first and last part of the time_series to extract the slope
            #last = epo_window[:,-1]
            #slope = last - first / window_size
            #slopes.append(slope)
            #find SD
            sd_array[:,i] = np.std(epo_window, axis=1)
            #find Kurtosis 
            kurt_array[:,i] = np.std(epo_window, axis=1)
       
        #slope_array= np.vstack(slopes).T    #init the array that takes all the means after the windowshift on all channels
        # add the features in the same vector, and turn them in a 1D matrix
        feat_vec_row = np.ravel(np.column_stack([mean_array,sd_array,kurt_array])) 
        # transpose them
        feat_vec = np.transpose(feat_vec_row[0:len_feat])
        # Assemble features into matrix X and predictions in the Y matrix
        X.append(feat_vec)
        Y = np.hstack((Y, row['Predict_help']))
    X = np.stack(X)
 
    
   
    #%% Extract features from the HRV

    # Sampling interval
    T= 1/raw_haemo_hrv.info['sfreq'] #sampling period
    ibi_series_ch=[] #init the ibi_series
    # Iterate over channels to extract interbeat intervals for each channel
    for ch_name in grp_ch_names_hbo: #change that, we should use hbr and hbo
         fnirs_1=raw_haemo_hrv.copy().pick_channels([ch_name]).pick_types(fnirs="hbr") # take the hbo of channels
         fnirs_1_ch=fnirs_1.get_data().flatten()                                       # flatten all the channels
         idx=np.arange(0, len(fnirs_1_ch))                                             # needed for plotting peaks in time series
         
         ### Detect peaks using AMPD algorithm. pks holds the peak index
         debug=True
         if debug: #If debug is set to True, pks_ampd, ZLSM, and l_scale are assigned the result of finding the peaks using the AMPD algorithm, with debug=True.
             pks_ampd, ZLSM, l_scale=ampd.find_peaks_original(fnirs_1_ch, debug=True)
         else:
             pks_ampd=ampd.find_peaks_original(fnirs_1_ch) #find the peaks here. 
         
         # Compute first difference of time series and its first difference
         first_diff=help_functions_stress.first_difference_series(fnirs_1_ch, T)
         max_first_diff=np.max(first_diff)
         
         # Use the AMPD to compute the IBIs
         ibi_ampd=help_functions_stress.ibi_timeseries(pks_ampd, T, len(fnirs_1_ch)) #compute IBIs from the peakes
         ibi_ampd=help_functions_stress.ibi_interpolation(ibi_ampd, method='polynomial', order=2) #interpolate
         ibi_series_ch.append(ibi_ampd)
    ibi_series=np.vstack(ibi_series_ch).mean(axis=0) # take average across channels

    # Turn the IBIs into features
    N_ibi = 30 # Number of samples to compute average IBI at the beginning and end of an epoch 
    ibi_list = []
    
    # Iterate over the events
    for index, row in df_events.iterrows():
        # Add 8 seconds to the onset time to account for the 5-8 second delay in the fNIRS signal
        idxs_evt = raw_haemo_hrv.time_as_index([row["Onset"] + 8, (row["Onset"] + row['Offset']) / 2, row['Offset']])
        # Compute the average IBI for the whole epoch, the beginning of the epoch, and the end of the epoch
        ibi_list.append([np.mean(ibi_series[idxs_evt[0]:idxs_evt[-1]]), np.mean(ibi_series[idxs_evt[0]: idxs_evt[0] + N_ibi]),
                         np.mean(ibi_series[idxs_evt[1]: idxs_evt[1] + N_ibi]),  np.mean(ibi_series[idxs_evt[-1] - N_ibi: idxs_evt[-1]])])
    
    # Concatenate the IBI features with the other features in X and X_ibi
    X = np.concatenate((X, ibi_list ), axis=1)
    X_ibi = np.concatenate((X, ibi_list ), axis=1)

    #%% Feature Filtering
    
    # Apply a ttest and cohen's d to det. which features are the most significant.
    import scipy.stats as stats


    #cohen's test
    def cohens_d_matrix(data1, data2):
        """Calculate Cohen's d for 2D matrices
            data1, data2 : numpy 2D arrays
        """
        d_matrix = []
        for i in range(data1.shape[1]):
            col1 = data1[:,i]
            col2 = data2[:,i]
            n1, n2 = len(col1), len(col2)
            mean1, mean2 = np.mean(col1), np.mean(col2)
            std1, std2 = np.std(col1), np.std(col2)
            pooled_std = np.sqrt(((n1 - 1) * std1 ** 2 + (n2 - 1) * std2 ** 2) / (n1 + n2 - 2))
            d = (mean1 - mean2) / pooled_std
            d_matrix.append(d)
        return d_matrix
    
    #p_values = t_test(X)    #find the ttest of all the features contrasting condition 1 to 2
    #significant_features = [i for i, p_value in enumerate(p_values) if p_value < 0.05] #find how much are stat. significants
    # group the 2 conditions of the X vector
    
    #the cohen's d takes our two conditions and find
    # Calculate Cohen's d for each significant feature
    

    #remove the cohen's d< abs(0.5)
    #Extract the latest part of the experiment only, both for X and Ys, so that we take conditions where the stress built up
    first_quartile_index = int(0.25 * X.shape[0])
    second_quartile_index = int(0.5 * X.shape[0])
    third_quartile_index = int(0.75 * X.shape[0])
    fourth_quartile_index = int(1 * X.shape[0])
    
    #take the indices of the conditions to keep
    keep_indices = np.concatenate((np.arange(first_quartile_index, second_quartile_index), np.arange(third_quartile_index, fourth_quartile_index)))
    #filter the X rows
    X = X[keep_indices, :]
    #filter Y vector
    Y = Y[keep_indices]
    
    #%% Fit the SVM model
    kf = KFold(n_splits=10, shuffle=False) #This is to ensure we keep the order of the original conditions not to break their correlations
    # Start by creating a k-fold loop
    for train_index, test_index in kf.split(X):
        acc_cv = []
        
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        
        #Apply the ttest abd  Cohen'd between the two condition, for the training set, and the testing set seperately
        #p_values = t_test(X_train,Y_train)    # find the t-test of all the features contrasting condition 1 to 2
        #significant_features = [i for i, p_value in enumerate(p_values) if p_value < 0.05] # find how many are statistically significant
        # filter out non-significant features before applying Cohen's d
        #X_ttrain = X_train[:, significant_features]
        cohen_d_train = cohens_d_matrix(X_train[Y_train==0,:], X_train[Y_train==1,:])
        #based on the cohen_d, drop the features that doesn't have significant differences between condition 1 and condition 2
        threshold = 0.5 
        # extract indices of features with cohen's d above threshold
        sf_cohen = np.where(np.abs(cohen_d_train) > threshold)[0]
        #significant_features_test = np.where(np.abs(cohen_d_test) > threshold)[0]    # extract significant features from X
        #removethe significant featurefrom both the Train and test
        X_train_new = X_train[:, sf_cohen]
        X_test_new = X_test[:, sf_cohen]
        
        #apply the PCA and cohen's d inside those sets
        X_train_new[np.isnan(X_train_new)] = 0
        X_train_new[np.isinf(X_train_new)] = 0
        X_test_new[np.isnan(X_test_new)] = 0
        X_test_new[np.isinf(X_test_new)] = 0
        #standartize the scale of the data
        scaler = StandardScaler()
        scaler.fit(X_train_new)
        X_train_std = scaler.transform(X_train_new)
        X_test_std = scaler.transform(X_test_new)
        #Apply PCA
        pca = PCA(n_components=10 )
        pca.fit(X_train_std)
        X_train_reduced = pca.transform(X_train_std)
        X_test_reduced = pca.transform(X_test_std)
        
        #now drop the 1st component, because it's usually the motion artefacts
        X_train_reduced = X_train_reduced[:, 1:]
        X_test_reduced = X_test_reduced[:, 1:]

        #fit the model
        
        #define the randomized parameters
        param_dist = {'C': uniform(loc=100, scale=500), 
                      'kernel': ['linear', 'rbf', 'poly'], 
                      'degree': [2], 
                      'class_weight': [None, 'balanced']}
    
        clf = svm.SVC(class_weight='balanced')
        random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=10,n_jobs=-1) #Randomize 
        random_search.fit(X_train_reduced, Y_train)
    
        # Get best parameters, score and model
        best_params = random_search.best_params_
        best_score = random_search.best_score_
        best_model = random_search.best_estimator_
        #Evaluate the accuracy of the best model
        y_pred = best_model.predict(X_test_reduced)
        acc_cv = metrics.accuracy_score(Y_test, y_pred)
        #append results
        results_subs.append(acc_cv)
        

    acc = np.mean(np.array(results_subs))
    
    # BOOTSTRAPING: Let's try to resample the data, and see if the accuracy we found is greater than the bootstrap
    n_samples = 1000  # number of bootstrap samples
    bootstrap_scores = []
    for i in range(n_samples):
        # generate bootstrap sample
        X_boot, y_boot = resample(X_train_reduced, Y_train)
        # calculate accuracy on observations
        y_bootpred = best_model.predict(X_boot)
        bootstrap_scores.append(metrics.accuracy_score(y_bootpred, Y_train))
    bootstrap_scores = np.array(bootstrap_scores)
    
    mean_bs = np.mean(bootstrap_scores)
    lower_ci = np.percentile(bootstrap_scores, 2.5)
    upper_ci = np.percentile(bootstrap_scores, 97.5)
    # Calculate 95% confidence interval
    error = np.array([[mean_bs - lower_ci], [upper_ci - mean_bs]])
    # Plot the mean with error bars
    acc_subs[sub_idx] = acc 
    acc_boot_subs[sub_idx] = mean_bs
    error_boot_subs[:,sub_idx] = error.reshape(2,)
   
#plot Accuracies for each subject near the bootstrap mean
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_title('Model Accuracy by Subject')
ax.set_xlabel('Subject')
ax.set_ylabel('Accuracy')

# Create bar plot for each subject
x = np.arange(n_subjects)
width = 0.35
rects1 = ax.bar(x - width/2, acc_subs, width, label='Accuracy')
rects2 = ax.bar(x + width/2, acc_boot_subs, width, yerr=error_boot_subs, label='Mean Bootstrapped Accuracy')
plt.xlim([-0.5, n_subjects-0.5])

# Add legend and show plot
ax.legend()
plt.show()

#This will generate a histogram with two bars per subject, one for the accuracy and one for the mean bootstrapped accuracy with error bars. You can modify the data generation code to use your own accuracy and mean bootstrapped accuracy arrays for multiple subjects.




 

    
            



#%% TEST the accuracies of the model
# Plot the accuracies of each subject
subject_names = ['06', '07', '08', '10','11','12','13','14','15','16','17'] 

plt.bar(subject_names, results_subs)
plt.xlabel('Subjects')
plt.ylabel('Accuracy')
plt.title('Accuracies of SVM for each subjects')
plt.ylim(0.4, 1)
plt.show()



"""
# Check how many conditions are used as a support vector
sv = best_model.support_vectors_
# plot the support vectors in a scatter plot
plt.scatter(sv[:, 0], sv[:, 1], s=100, c='black', marker='o')

# plot the rest of the data points
plt.scatter(X_train_reduced[:, 0], X_train_reduced[:, 1], c=y_train, s=30, cmap=plt.cm.Paired)
# add labels and show the plot
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Check the amounts of alphas that are zeros

plt.plot(best_model.dual_coef_[0], '-o')
plt.axhline(0, color='black', linestyle='--')
plt.xlabel('Coefficients Index')
plt.ylabel('Coefficient Value')
plt.title('SVM Dual Coefficients')
plt.show()  


non_zero_alphas = (best_model.dual_coef_ != 0).ravel()
#since all of the alphas were used, we have to change
#%%
"""

