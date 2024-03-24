from iao_toolbox import BrainvisionInfo, EEGRef, io, proc

# Creating parameter objects
info = BrainvisionInfo(
    montage="standard_1005",
    eog=('vEOG', 'hEOG', 'COS', 'ZYM', 'FDIL', 'FDIR', 'ECG'),
    preload=True,
    channel_types={'ECG': 'ecg', 'COS': 'emg', 'ZYM': 'emg', 'FDIL': 'emg', 'FDIR': 'emg'}
)

eegRef = EEGRef(
    ref_channels=['TP10'],
    iao_ref=True
)

# Specify data folder
folder = "./Data/emoio/09/EEG/Run_1.vhdr"

eeg_data = io.convert_brainvision_to_mne(folder, info)

print('----------------------')

tp10 = eeg_data.get_data()[21]
print(tp10)
print(tp10/2)

print('----------------------')
print(eeg_data.get_data()[0])
print(eeg_data.get_data()[0] - (tp10/2))
print(eeg_data.get_data()[1] - (tp10/2))

print('----------------------')

eeg_data = proc.set_eeg_ref(eeg_data, eegRef)

print(eeg_data.get_data()[0])
print(eeg_data.get_data()[1])
