"""
This io.py file contains methods and classes which convert several data-structures like brainvision or
customs to the mne-data-structure.
"""
from mne import io
from pathlib import Path, PureWindowsPath
from iao_toolbox import helpers


class BrainvisionInfo:
	"""
	Class Object which contains properties for loading brainvision
	into the mne-data-structure.

	Parameters
	----------
	montage:
	eog:
	misc:
	scale:
	preload:
	response_trig_shift:
	event_id:
	trig_shift_by_type:
	stim_channel:
	verbose:
	channel_types:

	Example
	--------
	info = BrainvisionInfo(
		montage="standard_1005",
		eog=('vEOG', 'hEOG', 'COS', 'ZYM', 'FDIL', 'FDIR', 'ECG'),
		preload=True,
		channel_types={'ECG': 'ecg', 'COS': 'emg', 'ZYM': 'emg', 'FDIL': 'emg', 'FDIR': 'emg'}
	)
	"""
	def __init__(self, eog=('HEOGL', 'HEOGR', 'VEOGb'), misc='auto', scale=1.0, preload=False,
				verbose=None, channel_types=None):
		self.info = dict(
			eog=eog, 
			misc=misc, 
			scale=scale, 
			preload=preload, 
			verbose=verbose
		)
		self.channel_types=channel_types
	
	# ~ def __init__(self, eog=('HEOGL', 'HEOGR', 'VEOGb'), misc='auto', scale=1.0, preload=False, verbose=None):
		# ~ self.info = dict(
			# ~ eog=eog, 
			# ~ misc=misc, 
			# ~ scale=scale, 
			# ~ preload=preload, 
			# ~ verbose=verbose
		# ~ )


def convert_brainvision_to_mne(path, info: BrainvisionInfo):
    """Load brainvision vhdr to mne and set channel types

    Loads brainvision vhdr data to mne by using BrainvisionInfo

    Parameters
    ----------
    path: str
        path to the .vdhr file
    info: BrainvisionInfo
        custom BrainvisionInfo class

    Returns
    -------
    data: mne.raw
        mne-data-structure

    Example
    -------
    info = BrainvisionInfo(
        montage="standard_1005",
        eog=('vEOG', 'hEOG', 'COS', 'ZYM', 'FDIL', 'FDIR', 'ECG'),
        preload=True,
        channel_types={'ECG': 'ecg', 'COS': 'emg', 'ZYM': 'emg', 'FDIL': 'emg', 'FDIR': 'emg'}
    )
    file_path = "./Data/brainvision/"
    data = io.convert_brainvision_to_mne(file_path, info)

    """

    data = io.read_raw_brainvision(path, **info.info)
    data.set_channel_types(mapping=info.channel_types)

    return data


""" 
def load_brainvision_of_subject(info: BrainvisionInfo, folder, sub_file, data_type, ending, list_of_files=None):

    raw_data_folder = PureWindowsPath(folder)
    raw_data_folder_path = Path(raw_data_folder).joinpath(sub_file, data_type)

    data = []

    # if list_of_files is None, load the whole folder
    if list_of_files is None:
        file_list = helpers.get_file_list(raw_data_folder_path, ending)
        file_list = helpers.sort_file_list(file_list)

        for file in file_list:
            file_path = raw_data_folder_path.joinpath(file)
            data.append(convert_brainvision_to_mne(file_path, info))

    # if list_of_files exist, load specific data from folder
    else:
        for file in list_of_files:
            file_path = raw_data_folder_path.joinpath(file)
            data.append(convert_brainvision_to_mne(file_path, info))

    return data
"""
