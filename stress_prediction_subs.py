"""
Created on Mon 22 16:31:46 2022

@author: Njeukwa Bounkeu, Ntsanyem
"""

# Import necessary libraries
import os
import numpy as np
import scipy as sc
import pandas as pd
import matplotlib.pyplot as plt
import sys
import mne, mne_nirs

# Append the directories of all the toolboxes
sys.path.append('C:/Users/ene Monsieur/OneDrive - Carl von Ossietzky Universität Oldenburg/Documents/OFFIS.de/fNIRS_data_analysis/Prasenjit_Datas/stress_analysis/stress_analysis')
sys.path.append('C:/Users/ene Monsieur/OneDrive - Carl von Ossietzky Universität Oldenburg/Documents/OFFIS.de/fNIRS_data_analysis/Prasenjit_Datas/stress_analysis/decoding_pire/')
sys.path.insert(1, 'C:/Users/ene Monsieur/OneDrive - Carl von Ossietzky Universität Oldenburg/Documents/OFFIS.de/fNIRS_data_analysis/Prasenjit_Datas/stress_analysis/decoding_pire/iaoToolboxCustom')

# Import custom modules
import proc
import nirs_help
import supporting_func, helpers, io, proc
import ampd
from proc import fNIRSreadParams, FilterParams
import help_functions_stress

# Import machine learning libraries
from sklearn.decomposition import PCA, FastICA
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold, permutation_test_score
from sklearn import svm, preprocessing, linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from sklearn import metrics
from sklearn.metrics import roc_curve, auc

# Info is necessary for extracting hbr signal
info = fNIRSreadParams(preload=True, verbose=None)

# Filter parameters to extract HRV
filterParams = FilterParams(
    l_freq=1,
    h_freq=1.9,
    method='fir',
    fir_design='firwin',
    phase='zero-double')

# Filter parameters to extract PFC activity
filterParams_all = FilterParams(
    l_freq=0.01,
    h_freq=0.7,
    method='fir',
    fir_design='firwin',
    phase='zero-double')

# Create a dictionary with paths and time intervals for each subject
path_dict = {
    '06_sub': ['2021-10-06_003', 'log_data', [(27.90, 142.47), (4630.59, 4732.28)]],
    '07_sub': ['2021-10-25_001', 'log_data', [(15.39, 161.10), (3700.05, 3825.94)]],
    '08_sub': ['2021-10-26_002', 'log_data', [(25.22, 112.73), (4377.42, 4456.86)]],
    '10_sub': ['2021-11-02_003', 'log_data', [(46.80, 120.55), (4351.66, 4474.44)]],
    '11_sub': ['2021-11-05_002', 'log_data', [(30.44, 152.23), (4687.38, 4832.41)]],
    '12_sub': ['2021-11-08_001', 'log_data', [(102.03, 222.54), (5005.38, 5130.31)]],
    '13_sub': ['2021-11-09_001', 'log_data', [(43.77, 191.85), (4302.88, 4526.49)]],
    '14_sub': ['2021-11-11_001', 'log_data', [(82.07, 205.04), (4901.64, 5023.67)]],
    '15_sub': ['2021-11-18_001', 'log_data', [(63.29, 220.40), (3851.89, 3983.02)]],
    '16_sub': ['2021-11-19_001', 'log_data', [(254.62, 351.98), (4602.82, 4612.69)]],
    '17_sub': ['2021-11-22_001', 'log_data', [(212, 284.69), (3901.64, 3950.72)]]
    # '18_sub': ['2021-11-23_001', 'log_data', [(63.29, 220.40), (3851.89, 3983.02)]],
    # '19_sub': ['2021-12-13_003', 'log_data', [(63.29, 220.40), (3851.89, 3983.02)]]
}

# Grouping of channels for stress prediction: frontal region/frontal channels for heart rate
SD_list = [[(2, 1), (3, 1), (2, 5), (6, 5), (7, 5)],
           [(1, 1), (1, 2), (1, 3), (3, 3), (4, 3), (3, 6), (4, 7), (7, 6), (8, 6), (8, 3), (8, 7), (9, 7)],
           [(5, 2), (5, 8), (4, 2), (9, 8), (10, 8), (10, 9)]]

# Convert source detector pairs into channel names (e.g., S1_D1 hbo)
grp_ch_names_hbo = help_functions_stress.flatten_list(help_functions_stress.create_grouping_optodes(SD_list, oxy_type='hbr'))

# FILTERING
baseFolder = 'C:/Users/ene Monsieur/OneDrive - Carl von Ossietzky Universität Oldenburg/Documents/OFFIS.de/fNIRS_data_analysis/Prasenjit_Datas/Data_PIRE_/'  # Load the base folder of all the subjects

results_subs = []  # Initialize the list to store results for each subject
fpr_list = []  # Initialize the list to store false positive rates
tpr_list = []  # Initialize the list to store true positive rates
thresholds_list = []  # Initialize the list to store thresholds
roc_auc_list = []  # Initialize the list to store ROC AUC values
results_subs = []  # Initialize the list to store results for each subject
fpr_list = []  # Initialize the list to store false positive rates
tpr_list = []  # Initialize the list to store true positive rates
thresholds_list = []  # Initialize the list to store thresholds
roc_auc_list = []  # Initialize the list to store ROC AUC values


for sub, value in path_dict.items():  # Loop over all subjects
    # Reading of fNIRS and behavioral data
    data_path = os.path.join(baseFolder, sub, value[0])
    raw_intensity = mne.io.read_raw_nirx(data_path, verbose=True, preload=True)
    sample_rate = raw_intensity.info['sfreq']

    # Convert to optical density and then to hemoglobin concentration
    raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)
    raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=0.1)

    # Filtering the HRV and artifacts out
    # Resample the data to match the sampling frequency mentioned in the paper
    raw_resampled = raw_haemo.resample(10)
    raw_haemo_hrv = proc.filter(raw_resampled.copy(), filterParams, 'bandpass')

    # Filter to extract only the HRV
    raw_haemo = raw_haemo.copy().filter(0.05, 0.7, h_trans_bandwidth=0.2,
                                        l_trans_bandwidth=0.02)

    # Find the base average
    base_inter = value[2]  # value is part of the path for the subject
    baseline_1 = supporting_func.get_data(raw_haemo.copy(), 'hbr', base_inter[0]).mean(axis=1)
    baseline_2 = supporting_func.get_data(raw_haemo.copy(), 'hbr', base_inter[1]).mean(axis=1)
    base_avg = np.mean([baseline_1, baseline_2], axis=0)[:, np.newaxis]  # Take the average of baseline data
    del baseline_1, baseline_2

    # Load the events excel file to pick the epochs from the data
    df_events = nirs_help.read_excel_sheets(os.path.join(baseFolder, sub, value[0], 'events.xlsx'))  # Loading events
    df_events['Predict_help'] = df_events['Condition'].apply(lambda x: 1 if x == 'Scenario_2' else 0)  # Creating the labels for scenarios
    cols = ['Onset', 'Offset']

    # Extract the features
    X = []  # Initialize feature matrix X
    Y = []
    X_ibi = []  # Initialize matrix for only the IBIs
    nFeat = 2  # Define how many parts of the data you want (e.g., to find the mean of 2 parts of the data)
    cols = ['Onset', 'Offset']  # Define the columns to extract epochs from events.xlsx

    # FEATURE EXTRACTION
    for index, row in df_events.iterrows():  # Iterate through each epoch
        # Extract the time-series*channels of each epoch, and keep the channels of the PFC only
        epo_data = []  # DataFrame of the channels*timeseries
        ch_data = []  # DataFrame of each channel
        t = (row[cols[0]] + 10, row[cols[0]] + 30)  # Define the range of time to extract fNIRS data
        for ch_name in grp_ch_names_hbo:  # Loop through the ROI channels (the PFC) - This will discard unuseful channels
            ch_data_hbr = raw_haemo.copy().pick_channels([ch_name]).pick_types(fnirs='hbr')  # Extract the hbr only
            ch_data = ch_data_hbr.crop(tmin=t[0], tmax=t[1], include_tmax=True).get_data()  # Extract the hbr only
            if len(epo_data) == 0:  # Initialize the length of the matrix so that the other ch_data can fit
                epo_data = ch_data
            else:
                epo_data = np.concatenate((epo_data, ch_data), axis=0)  # Stack the chosen channels in epo_data

        # Initialize the length of the time series and channels of the epo_data
        len_time = len(epo_data[0])  # Length of the time series that will take all the means
        len_channels = len(epo_data)
        window_size = len_time // nFeat  # Define the size of the block of time series we will extract features from
        len_feat = len_channels * nFeat + len_channels * nFeat  # Length of the channels

        # Initialize the data frame of the features we will extract: Mean, Slope, SD, Kurtosis...
        mean_array = np.zeros((len_time // window_size, len_channels)).T  # Initialize the array that takes all the means after the window shift on all channels
        slopes = []  # Initialize the slope
        sd_array = np.zeros((len_time // window_size, len_channels)).T  # Initialize the array that takes all the standard deviations after the window shift on all channels
        kurt_array = np.zeros((len_time // window_size, len_channels)).T  # Initialize the array that takes all the kurtosis values after the window shift on all channels

        # Loop through parts of the time series to extract data from them
        for i in range(len_time // window_size):  # Define the portions of the time series to extract
            start = i * window_size  # Define start and end of the block of time series
            end = (i + 1) * window_size
            epo_window = epo_data[:, start:end]  # Extract that part
            mean_array[:, i] = np.mean(epo_window, axis=1)  # Calculate the mean of that block

            # Find the mean slopes
            first = epo_window[:, 0]  # Define first and last part of the time series to extract the slope
            last = epo_window[:, -1]
            slope = last - first / window_size
            slopes.append(slope)
            # Find SD
            sd_array[:, i] = np.std(epo_window, axis=1)
            # Find Kurtosis
            kurt_array[:, i] = np.std(epo_window, axis=1)

        slope_array = np.vstack(slopes).T  # Initialize the array that takes all the slopes after the window shift on all channels
        # Add the features in the same vector, and turn them into a 1D matrix
        feat_vec_row = np.ravel(np.column_stack([mean_array, slope_array, sd_array, kurt_array]))
        # Transpose them
        feat_vec = np.transpose(feat_vec_row[0:len_feat])
        # Assemble features into matrix X and predictions in the Y matrix
        X.append(feat_vec)
        Y = np.hstack((Y, row['Predict_help']))
    X = np.stack(X)

    # Extract features from the HRV
    # Sampling interval
    T = 1 / raw_haemo_hrv.info['sfreq']  # Sampling period
    ibi_series_ch = []  # Initialize the ibi_series
    # Iterate over channels to extract interbeat intervals for each channel
    for ch_name in grp_ch_names_hbo:  # Change that, we should use hbr and hbo
        fnirs_1 = raw_haemo_hrv.copy().pick_channels([ch_name]).pick_types(fnirs="hbr")  # Take the hbo of channels
        fnirs_1_ch = fnirs_1.get_data().flatten()  # Flatten all the channels
        idx = np.arange(0, len(fnirs_1_ch))  # Needed for plotting peaks in time series

        # Detect peaks using AMPD algorithm. pks holds the peak index
        debug = True
        if debug:  # If debug is set to True, pks_ampd, ZLSM, and l_scale are assigned the result of finding the peaks using the AMPD algorithm, with debug=True.
            pks_ampd, ZLSM, l_scale = ampd.find_peaks_original(fnirs_1_ch, debug=True)
        else:
            pks_ampd = ampd.find_peaks_original(fnirs_1_ch)  # Find the peaks here.

        # Compute first difference of time series and its first difference
        first_diff = help_functions_stress.first_difference_series(fnirs_1_ch, T)
        max_first_diff = np.max(first_diff)

        # Use the AMPD to compute the IBIs
        ibi_ampd = help_functions_stress.ibi_timeseries(pks_ampd, T, len(fnirs_1_ch))  # Compute IBIs from the peaks
        ibi_ampd = help_functions_stress.ibi_interpolation(ibi_ampd, method='polynomial', order=2)  # Interpolate
        ibi_series_ch.append(ibi_ampd)
    ibi_series = np.vstack(ibi_series_ch).mean(axis=0)  # Take the average across channels

    # Turn the IBIs into features
    N_ibi = 30  # Number of samples to compute the average IBI at the beginning and end of an epoch
    ibi_list = []

    # Iterate over the events
    for index, row in df_events.iterrows():
        # Add 8 seconds to the onset time to account for the 5-8 second delay in the fNIRS signal
        idxs_evt = raw_haemo_hrv.time_as_index([row["Onset"] + 8, (row["Onset"] + row['Offset']) / 2, row['Offset']])
        # Compute the average IBI for the whole epoch, the beginning of the epoch, and the end of the epoch
        ibi_list.append([np.mean(ibi_series[idxs_evt[0]:idxs_evt[-1]]),
                         np.mean(ibi_series[idxs_evt[0]: idxs_evt[0] + N_ibi]),
                         np.mean(ibi_series[idxs_evt[1]: idxs_evt[1] + N_ibi]),
                         np.mean(ibi_series[idxs_evt[-1] - N_ibi: idxs_evt[-1]])])

    # Concatenate the IBI features with the other features in X and X_ibi
    X = np.concatenate((X, ibi_list), axis=1)
    X_ibi = np.concatenate((X, ibi_list), axis=1)

    # Feature Filtering
    # Apply a t-test and Cohen's d to determine which features are the most significant.
    import scipy.stats as stats

    # t-test
    def t_test(X):
        p_values = []
        for i in range(X.shape[1]):
            feature = X[:, i]
            t_test, p_value = stats.ttest_ind(feature[Y == 0], feature[Y == 1])
            p_values.append(p_value)
        return p_values

    # Cohen's test
    def cohens_d_matrix(data1, data2):
        """Calculate Cohen's d for 2D matrices
           data1, data2 : numpy 2D arrays
        """
        d_matrix = []
        for i in range(data1.shape[1]):
            col1 = data1[:, i]
            col2 = data2[:, i]
            n1, n2 = len(col1), len(col2)
            mean1, mean2 = np.mean(col1), np.mean(col2)
            std1, std2 = np.std(col1), np.std(col2)
            pooled_std = np.sqrt(((n1 - 1) * std1 ** 2 + (n2 - 1) * std2 ** 2) / (n1 + n2 - 2))
            d = (mean1 - mean2) / pooled_std
            d_matrix.append(d)
        return d_matrix

    p_values = t_test(X)  # Find the t-test of all the features contrasting condition 1 to 2
    significant_features = [i for i, p_value in enumerate(p_values) if p_value < 0.05]  # Find how many are statistically significant

    # Filter out non-significant features before applying Cohen's d
    X_significant = X[:, significant_features]

    # Calculate Cohen's d for each significant feature
    condition1 = X_significant[Y == 0]
    condition2 = X_significant[Y == 1]
    n1, n2 = len(condition1), len(condition2)
    var1, var2 = np.var(condition1, ddof=1), np.var(condition2, ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    cohen_d_values = []
    for feature in range(X_significant.shape[1]):
        mean_diff = np.mean(condition2[:, feature]) - np.mean(condition1[:, feature])
        cohen_d = mean_diff / pooled_sd
        cohen_d_values.append(cohen_d)

    # Select the most relevant features based on Cohen's d
    threshold = 0.5
    X_relevant = X_significant[:, np.abs(cohen_d_values) > threshold]

    # Extract the latest part of the experiment only, both for X and Y, so that we take conditions where the stress built up
    first_quartile_index = int(0.25 * X.shape[0])
    second_quartile_index = int(0.5 * X.shape[0])
    third_quartile_index = int(0.75 * X.shape[0])
    fourth_quartile_index = int(1 * X.shape[0])

    # Take the indices of the conditions to keep
    keep_indices = np.concatenate((np.arange(first_quartile_index, second_quartile_index),
                                   np.arange(third_quartile_index, fourth_quartile_index)))
    # Filter the X rows
    X = X[keep_indices, :]
    # Filter Y vector
    Y = Y[keep_indices]

    # Features filtering: We use PCA on the training set to capture the most variability from the data
    # Remove the empty values
    from sklearn.preprocessing import StandardScaler
    X_copy = X.copy()
    X[np.isnan(X)] = 0
    X[np.isinf(X)] = 0

    # Do PCA on the X_train
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=109)

    # Standardize the scale of the data
    scaler = StandardScaler()
    # Fit the scaler on the training set
    scaler.fit(X_train)
    # Transform both training and testing sets
    X_train_std = scaler.transform(X_train)
    X_test_std = scaler.transform(X_test)

    # Apply PCA
    pca = PCA(n_components=10)
    pca.fit(X_train_std)
    X_train_reduced = pca.transform(X_train_std)
    X_test_reduced = pca.transform(X_test_std)

    # Drop the 1st component, as it's usually the motion artifacts
    X_train_reduced = X_train_reduced[:, 1:]
    X_test_reduced = X_test_reduced[:, 1:]

    # SVM analysis
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import uniform
    from sklearn import metrics
    from sklearn.metrics import roc_curve, auc

    param_dist = {'C': uniform(loc=100, scale=500),
                  'kernel': ['linear', 'rbf', 'poly'],
                  'degree': [2],
                  'class_weight': [None, 'balanced']}

    clf = svm.SVC(class_weight='balanced')
