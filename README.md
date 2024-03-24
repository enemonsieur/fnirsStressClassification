# fNIRS Data Analysis for Stress Prediction

## Overview
This repository hosts `stress_prediction_subs.py`, a Python script for processing and analyzing functional Near-Infrared Spectroscopy (fNIRS) data to predict stress levels. The analysis assesses heart rate variability (HRV) and prefrontal cortex activity, targeting stress level differentiation in power grid system tasks.

### Data Reading and Preprocessing
- **Acquisition:** Utilizes `fNIRSreadParams` from the `proc` module to ingest data from the specified path.
- **Filtering:** `FilterParams` in the `proc` module removes unwanted frequencies.

### Peak Detection
- Implements AMPD, M2D, and S1 functions for identifying heartbeats in time series data.

### Interbeat Interval (IBI) Calculation
- Calculates IBIs to assess heart rate variability, focusing on the average IBI across epochs to identify temporal variations, adds it into the feature matrix

### Statistical Analysis and Feature Reduction
- Conducts t-tests and Cohen's d-tests to narrow down the feature set, highlighting significant stress indicators.
- Uses Principal Component Analysis (PCA) for reducing dimensions, prepping the data for machine learning.

### Machine Learning for Stress Classification
- Applies Support Vector Machine (SVM) to differentiate stress levels using combined HRV and fNIRS data, analyzing task-induced stress variations.
- Feature reduction and SVM classification are based on a 10-fold cross-validation method to ensure effective training and testing.

### Output Generation
- Produces IBI values and statistical analyses, usually saved in an Excel file for IBI data and a text file for statistics.

### Data Visualization
- `plot_timeseries_marker` function visualizes time series with detected peaks, aiding in the interpretation of heart rate fluctuations and peak detection.

## Methodology
The project examined stress in power grid tasks, Balance Group Management (BGM) and Fault Group Management (FM), predicting higher stress in FM due to its time-sensitive nature. Extensive signal processing and statistical analysis provided a basis for identifying stress-related physiological changes.

### Signal Processing and Classification
- Using the MNE library for preprocessing, the data underwent baseline correction and filtering, focusing on prefrontal cortex signals.
- Stress classification employed SVM, post-PCA for feature optimization, assessing stress levels from HRV and fNIRS signals.

## Results
- The SVM classifier offered a nuanced view of stress levels across BGM and FM tasks, reflecting substantial inter-subject variation and the complex nature of task-related stress.
- NASA-TLX findings showed no marked difference in stress perception between tasks, underscoring the challenge of subjective stress evaluation.

## Getting Started
1. **Setup:** Prepare a Python environment with libraries like numpy, scipy, pandas, matplotlib, mne, mne_nirs, and ampd.
2. **Data Configuration:** Set the data path in the script to your fNIRS data files.
3. **Execution:** Run the script to begin the data analysis process, with outcomes available in the output directory.

## Conclusion
`stress_prediction_subs_adapted_ePu.py` offers an advanced method for fNIRS-based stress prediction, integrating signal processing and machine learning to classify stress levels intricately. This process illuminates the intricacies of stress measurement in practical tasks, advocating for customized assessment approaches.

