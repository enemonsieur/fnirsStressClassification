# fNIRS Data Analysis for Stress Prediction

## Overview
This repository hosts `stress_prediction_subs.py`, a Python script for processing and analyzing functional Near-Infrared Spectroscopy (fNIRS) data to classify stress levels. The analysis assesses heart rate variability (HRV) and prefrontal cortex activity, targeting stress level differentiation in power grid system tasks.

### fNIRS Data Analysis for Stress Prediction

#### Data Loading and Preprocessing
- **Load Data**: `mne.io.read_raw_nirx` loads fNIRS data.
- **Optical Density Conversion**: `mne.preprocessing.nirs.optical_density` converts raw fNIRS data to optical density.
- **Hemoglobin Concentration Conversion**: `mne.preprocessing.nirs.beer_lambert_law` changes optical density to hemoglobin concentration.
- **Filtering**: `proc.filter` isolates HRV and PFC activity, critical for understanding stress response.

#### Feature Extraction
- **Time-series Analysis**: Extracts features like mean, slope, standard deviation, kurtosis directly from the fNIRS time series.
- **Peak Detection**: `ampd.find_peaks_original` identifies heartbeats to calculate interbeat intervals (IBIs).

#### Feature Selection
- **Statistical Testing**: `scipy.stats.ttest_ind` identifies significant features from the time-series data.
- **Effect Size Measurement**: Custom function `cohens_d_matrix` calculates Cohen's d to measure the effect size, helping prioritize features.

#### Data Preprocessing for Classification
- **Data Splitting**: `sklearn.model_selection.train_test_split` divides data into training and testing sets.
- **Standardization**: `sklearn.preprocessing.StandardScaler` normalizes the feature data.
- **Principal Component Analysis**: `sklearn.decomposition.PCA` reduces feature dimensions, focusing on most informative aspects.

#### Classification
- **SVM Modeling**: `sklearn.svm.SVC` is used for stress level prediction, identifying patterns in physiological responses.
- **Hyperparameter Optimization**: `sklearn.model_selection.RandomizedSearchCV` finds optimal parameters for the SVM classifier, ensuring robust performance.


## Methodology
The project examined stress in power grid tasks, Balance Group Management (BGM) and Fault Group Management (FM), predicting higher stress in FM due to its time-sensitive nature. Extensive signal processing and statistical analysis provided a basis for identifying stress-related physiological changes.

### Signal Processing and Classification
- Using the MNE library for preprocessing, the data underwent baseline correction and filtering, focusing on prefrontal cortex signals.
- Stress classification employed SVM, post-PCA for feature optimization, assessing stress levels from HRV and fNIRS signals.

## Results
- The SVM classifier show a 70% accuracy

## Getting Started
1. **Setup:** Prepare a Python environment with the libraries: numpy, scipy, pandas, matplotlib, mne, mne_nirs, and ampd.
2. **Data Configuration:** Set the data path in the script to your fNIRS data files.
3. **Execution:** We used Spyder to run the script

