"""
The helpers.py file contains methods and classes which don't belong to any other categories.

"""

import re
import os, pathlib
from pathlib import Path, PureWindowsPath
from typing import List
import scipy.io as scio
import numpy as np
import mne
import matplotlib.pyplot as plt
#from math import log10, floor

def sort_file_list(file_list: List[str]) -> List[str]:
    """Sort a list of file names

    This method will solve the problems with file names like Run_9 and Run_10.
    With common sort-methods, Run_10 would be come first. This method will sort
    file list names the right way.

    Disclaimer: It Only works with file names, which are almost the same like
                Run_1, Run_2, Run_3, Run_4, ..., Run_10

    Parameter
    ---------
    file_list: List[str]
        List of strings

    Returns
    -------
    new_file_list: List[str]
        List of strings

    Example
    -------
    file_list = sort_file_list(file_list)

    """

    new_file_list = []
    num_list = []

    # Regular Expression to get a number of an string
    nums = re.compile(r"[+-]?\d+(?:\.\d+)?")

    # Add numbers as integers to num_list
    for item in file_list:
        num_list.append(int(nums.search(item).group(0)))

    # Create a dict
    file_dict = dict(zip(num_list, file_list))

    # Sort dict depending key values
    for k, v in sorted(file_dict.items()):
        new_file_list.append(v)

    return new_file_list


def get_file_list(path, ending: str=None, crawl=False, sort=False, exclude=None) -> List[str]:
    """Get a list of file names or folders

    This helper method will return a list of file names in a specific folder
    and with a specific file ending.

    Parameter
    ---------
    path: str
        path where the folder is located
    ending: str
        file ending of the files, which should be read. If ending is none, it will return all files
        in the specific folder
    crawl: boolean
        this boolean determines if the this method should crawl through the directories or not.
        This is especially helpful if you want the folder names
    sort: boolean
        sort the list
    exclude: list
        contains file strings which should exclude/remove from the file_list

    Returns
    -------
    file_list: List[str]
        list of file names

    Example
    -------
    raw_data_path = helpers.get_raw_path(folder, ('09', 'eeg'))
    file_list = helpers.get_file_list(raw_data_path, '.vhdr', sort=True)

    """
    file_list = []

    # Get the file names of the folder
    for file in os.listdir(path):

        # Go inside directory if this file is a directory
        if crawl and os.path.isdir(path.joinpath(file)):
            file_list += get_file_list(path.joinpath(file), ending=ending)

        # Otherwise proceed with reading the files
        else:

            # Pick the files with the given ending
            if ending is not None:
                if file.endswith(ending):
                    file_list.append(str(file))

            # Otherwise pick all the files in the specific path
            else:
                file_list.append(str(file))

    # Exclude items in file_list, depending on the exclude list
    if exclude is not None:
        for ex in exclude:
            if ex in file_list:
                file_list.remove(ex)

    # Sort the list
    if sort:
        file_list = sort_file_list(file_list)

    return file_list


def get_raw_path(root_path, join_path):
    """Get the raw path

    Get a path by joining individual paths

    Parameter
    ---------
    root_path: str
    join_path: tuple
        containing individual paths

    Returns
    -------
    raw_data_folder_path: str
        joined path string

    Example
    -------
    raw_data_path = helpers.get_raw_path(folder, ('09', 'eeg'))

    """

    raw_data_folder = PureWindowsPath(root_path)
    raw_data_folder_path = Path(raw_data_folder).joinpath(*join_path)
    
    return raw_data_folder_path


def read_behave_mat(path, readable=False):
	"""Read SAM mat files

	Read behavioral SAM mat files and convert this to a readable numpy array

	Parameter
	---------
	path: str
		path of the .mat file
	readable: boolean
		If true, it will return a clean non-nested numpy array. Otherwise it will
		return the numpy array as it is

	Returns
	-------
	new_list/behave_list: np.ndarray
		converted np array of the loaded .mat file

	Examples
	--------
	path = './Data/09/Behavioral/SAM_run4.mat'
	mat_file = helpers.read_behave_mat(path, readable=True)

	"""
	# ~ exp_file_s= scio.loadmat(path, squeeze_me=True)
	# ~ sam_res_s=exp_file_s['samRes']
	# ~ print("sam_res_s: ", sam_res_s)
	# Load .mat file
	exp_file = scio.loadmat(path)
	sam_res = exp_file['samRes'][0]
	
	# Unravel unnecessary nested list
	behave_list = []
	for sam in sam_res:
		behave_list.append(sam[0][0])

	# Convert to numpy array
	behave_list = np.asarray(behave_list)
	
	# Convert to a much cleaner readable array
	new_list = []
	for li in behave_list:
		row_list = []
		for i in range(len(li)):
			if isinstance(li[i][0], np.ndarray):
				row_list.append(li[i][0][0])
			else:
				row_list.append(li[i][0])
			if i in (1, 3, 5):
				if len(str(row_list[-1])) > 1:
					row_list[-1] = row_list[-1][:-1]
					
		new_list.append(row_list[:])
		del row_list[:]
	# Return new_list if readable is true, otherwise return behave_list
	return np.asarray(new_list) if readable else behave_list


def get_subject_list(root):
    """Returns a list of subjects

    Parameter
    ---------
    root: Path
        path of the directory, where the subject folder is

    Returns
    -------
    file_list: List[str]
        returns a list of subjects

    Examples
    --------
    path = ./Data/emoio
    subject_list = helpers.get_subject_list(root)

    """
    return get_file_list(root, crawl=False)


def replace_emoio_event_markers(eeg_data, behave_data):
    """Replace emoio event markers

    With the help of the behave_data, the number 3 marker events will be replaces with the
    corresponding states like pos: 15, neg: 16 and neu: 17

    Parameter
    ---------
    eeg_data: mne.raw
        mne data object with the event markers that should be replaced
    behave_data: numpy array
        SAM .mat file which contains the states

    Returns
    -------
    eeg_data: mne.raw
        same mne raw object, but with replaced events

    Examples
    --------

    """
    events = mne.find_events(eeg_data)

    # Get indices of events with 3
    event_index = []
    for i in range(len(events)):
        if events[i][2] == 3:
            event_index.append(i)

    # Define dictionary
    state_dict = {
        'pos': 15,
        'neg': 16,
        'neu': 17
    }

    for j in range(len(event_index)):
        events[event_index[j]][2] = state_dict[behave_data[j, 0][0:3]]

    eeg_data.add_events(events, replace=True)

    return eeg_data


def get_run_list(raw_fnirs_path, excludeBaseline=False) -> List[str]:
	"""
	This helper method will return the name of actual run folders (Run_3 to Run_12) for a subject. 
	Usually, there are three extra run files for each subject in EMOIO dataset.
	
	Run_1: is a pre-experiment resting state recording with alternating cycles of 15sec Eyes open (EO) and 15sec of Eyes closed (EC). 
	If you check the corresponding marker file there will be two different marker events – 2 (EC) and 4 (EO). The run time should be 360s ~ 6min
	
	Run_2: is an example run, with only a few pics to illustrate the participants the trial procedure, so that familiarize with procedure 
	and the SAM-answers provided- You can ignore this file
	
	Run_13: is a post-experiment resting state recording again with the same alternating cycles of 15sec EO and 15sec of Eyes EC. 
	Same marker events as in the pre-resting state – 2 (EC) and 4 (EO). The run time should be 360s ~ 6min
	
	
	Parameters:
	----------
	raw_fnirs_path: str
		Path where fNIRS data are kept for each subject
	excludeBaseline: Boolean
		if True: returns the list of actual runs (Run_3 to Run_12) folders
		if False: returns all the folder
	
	Returns:
	-------
	Run folder lists[str]
	
	"""
	runFoldList=[]
	if os.path.isdir(raw_fnirs_path):
		runFoldList=sorted(next(os.walk(raw_fnirs_path))[1]) # get the name of all run folders for a particular subject
		
		if excludeBaseline is True:
			runFoldList.pop(-1) # remove run 13 i.e., resting state condition run folder
			runFoldList.pop(1) # remove test run folder
			runFoldList.pop(0) # remove baseline folder

	return runFoldList



def get_baseline_folder(raw_fnirs_path):
	"""
	This helper method will return the baseline run folder for a subject. 
	
	Run_1: is a pre-experiment resting state recording with alternating cycles of 15sec Eyes open (EO) and 15sec of Eyes closed (EC). 
	If you check the corresponding marker file there will be two different marker events – 2 (EC) and 4 (EO). The run time should be 360s ~ 6min
	
	Parameters:
	----------
	raw_fnirs_path: str
		Path where fNIRS data are kept for each subject
	
	Returns:
	-------
	baseline folder [str]
	"""
	runFoldList=[]
	if os.path.isdir(raw_fnirs_path):
		runFoldList=sorted(next(os.walk(raw_fnirs_path))[1]) # get the name of all run folders for a particular subject
	return runFoldList[0]

# ~ def onclick(event):
	# ~ global ix, iy
	# ~ ix, iy=event.xdata, event.ydata
	# ~ print('t1=%f, t2=%f' %(ix,iy))
	# ~ fig.canvas.mpl_disconnect(cid)

def get_time_instances(data):
	"""
	This helper method will return the time instances of two clicks from a figure. 
	
	Run_1: is a pre-experiment resting state recording with alternating cycles of 15sec Eyes open (EO) and 15sec of Eyes closed (EC). 
	If you check the corresponding marker file there will be two different marker events – 2 (EC) and 4 (EO). The run time should be 360s ~ 6min
	So
	
	Parameters:
	----------
	data: str
	Data to be plotted. Click two times in the figure to get the time insances (x axis) of two clicks
	
	Returns:
	-------
	List of two time instances
	"""
	
	global coords
	coords=[]
	def onclick(event):
		coords.append((event.xdata, event.ydata))
		
		if len(coords) == 2:
			fig.canvas.mpl_disconnect(cid)

		#return coords	
	
	fig=data.plot(n_channels=len(data.ch_names), title='Select baseline data between two clicks', duration=500, show_scrollbars=True) # show=True, block=True, 
	cid=fig.canvas.mpl_connect('button_press_event', onclick)
	plt.show()
	
	time_inter=[]
	if len(coords) == 2:
		# ~ print('coords', coords)
		time_inter.append(coords[0][0]) # Extract only time instants
		time_inter.append(coords[1][0])
		if time_inter[0] > time_inter[1]: # Swap if time value of first click is greater than time value of later click
			time_inter[0], time_inter[1]=time_inter[1], time_inter[0]
		
	return time_inter


def get_baseline_data(data_raw, time_inter):
	"""
	This helper method will return baseline data between two time instances.
	
	Run_1: is a pre-experiment resting state recording with alternating cycles of 15sec Eyes open (EO) and 15sec of Eyes closed (EC). 
	If you check the corresponding marker file there will be two different marker events – 2 (EC) and 4 (EO). The run time should be 360s ~ 6min
	
	Parameters:
	----------
	data: 
	fNIRS or EEG data for baseline calculation
	
	time_inter:
	time interval for considering baseline correction
	Returns:
	-------
	baseline fNIRS or EEG signal
	"""
	data=data_raw.get_data() # get the data from raw instance
	
	i1, i2=data_raw.time_as_index(time_inter[0])[0], data_raw.time_as_index(time_inter[1])[0]
	return np.mean(data[:, i1:i2], axis=1) # average data for each channel in between the time intervals 
	
def baseline_correction(epo, baseline_data):
	"""
	This helper method will apply customize baseline correction in epochs.
	
	Run_1: is a pre-experiment resting state recording with alternating cycles of 15sec Eyes open (EO) and 15sec of Eyes closed (EC). 
	If you check the corresponding marker file there will be two different marker events – 2 (EC) and 4 (EO). The run time should be 360s ~ 6min
	
	Parameters:
	----------
	data: 
	fNIRS or EEG data for baseline calculation
	
	Returns:
	-------
	baseline fNIRS or EEG signal
	"""
	
	if len(baseline_data) == 0:
		print('No Baseline data. Baseline correction failed')
		return epo
	
	# ~ print('len(baseline_data):', len(baseline_data))
	# ~ print('len(epo.get_data()): ', len(epo.get_data()))
	# ~ print('epo.get_data().shape', epo.get_data().shape)
	
	# This portion of the code is right but it prints a lot text in the display screen
	# ~ for e in range(len(epo.get_data())):
		# ~ for i in range(len(epo.ch_names)):
			# ~ epo.get_data()[e,i,:]=epo.get_data()[e,i,:]-baseline_data[i] # Epoch data shape: (#epochs, #ch, #sample numbers in a epoch)
	# ~ return epo
	
	data=epo.get_data()
	
	print("len(epo.ch_names): ", len(epo.ch_names))
	
	for e in range(len(data)): # for each epoch
		for i in range(len(epo.ch_names)): # for each channel
			data[e,i,:]=data[e,i,:]-baseline_data[i] # Epoch data shape: (#epochs, #ch, #sample numbers in a epoch)
	
	epo._data[:,:,:]=data
	return epo
	
def replace_emoio_fnirs_event_markers(events, behave_data):
	"""Replace emoio event markers

    With the help of the behave_data, the number 3 marker events will be replaces with the
    corresponding states like pos: 15, neg: 16 and neu: 17

    Parameter
    ---------
    event: 
        event markers that should be replaced
    behave_data: numpy array
        SAM .mat file which contains the states

    Returns
    -------
    event: mne.raw
        same event file but with modified labelling

    Examples
    --------

	"""
	
	# Get indices of events with 3
	event_index = []
	for i in range(len(events)):
		if events[i][2] == 3:
			event_index.append(i)

	# Define dictionary
	state_dict = {
		'pos': 15,
		'neg': 16,
		'neu': 17
		}
	# get the keyword (behave_data[j, 0][0:3]) like pos, neg, neu and match corresponding number (state_dict['neu']=17) from dictionary state_dict
	for j in range(len(event_index)):
		events[event_index[j]][2] = state_dict[behave_data[j, 0][0:3]] 

	#eeg_data.add_events(events, replace=True)

	return events

def find_exp(number) ->int:
	return floor(np.log10(abs(number)))



def normalize_data(raw):
	"""
	Normalize(divide each channel by its std) fNIRS data along each channel.

    Parameter
    ---------
    raw: 
        fNIRS or eeg data that need to normalize along each channel
    
    Returns
    -------
    raw_norm
        normalized raw data

    Examples
    --------

	"""
	
	
	# Get the data
	#anno=raw.annotations
	data=raw.get_data()
	std_val=np.std(data, axis=1)
	
	# Remove any e* (let say e-6) component from the standard deviation value. Because the exponent component 
	# scale up (fNIRS_data/*e-6) the fNIRS data while dividing each channel by its stanard deviation
	#sc_val=np.floor(np.log10(abs(std_val))) # log10(-4.27e-3)--> -3 i.e., extract the exponent part of a number
	#std_val_sc=std_val*np.power(10, -sc_val) # (5.653e-7)*(1*+7)=5.65 i.e., remove e-* component from a number
	
	data_norm=data/std_val[:,np.newaxis]
	
	# ~ raw_norm=mne.io.RawArray(data_norm, raw.info).set_annotations(raw.annotations)	
	# ~ np.savetxt(os.path.join('/data2/pdhara/work/output/EMOIO/fNIRS_result/raw_data/', 'std_val.txt'), std_val)
	
	raw._data[:,:]=data_norm #*10**(-6)
	
	return raw


# ~ def saveFigure(fig_list=[], flag_save=False, fig_save_path=None, add_path=None, name_list=[], flag_fig_close=False):
	# ~ if flag_save is False:
		# ~ return
	
	# ~ if add_path is not None:
		# ~ fig_path_final=os.path.join(fig_save_path, add_path)
	# ~ else:
		# ~ fig_path_final=fig_save_path
	
	# ~ pathlib.Path(fig_path_final).mkdir(parents=True, exist_ok=True)
	
	# ~ if fig_list and len(fig_list)==len(name_list):
		# ~ for i in range(len(fig_list)):
			# ~ fig_name=name_list[i]+'.pdf'
			# ~ print("figure_name: ", fig_name)
			# ~ fig_list[i].savefig(os.path.join(fig_path_final, str(name_list[i])+'.pdf'))
			# ~ fig_list[i].savefig(os.path.join(fig_path_final, str(name_list[i])+'.eps'))
			# ~ fig_list[i].savefig(os.path.join(fig_path_final, str(name_list[i])+'.png'))
			# ~ if flag_fig_close is True:
				# ~ plt.close(fig_list[i])


def saveFigure(fig_list=[], flag_save=False, fig_save_path=None, add_path=None, name_list=[], flag_fig_close=False):
	if flag_save is False:
		return
	
	if add_path is not None:
		fig_path_final=os.path.join(fig_save_path, add_path)
	else:
		fig_path_final=fig_save_path
	
	pathlib.Path(fig_path_final).mkdir(parents=True, exist_ok=True)
	
	if fig_list and len(fig_list)==len(name_list):
		for fig, name in zip(fig_list, name_list):
			#fig=fig_list[i]
			#print("figure_name: ", name)
			fig.savefig(os.path.join(fig_path_final, name+'.pdf'), bbox_inches='tight')
			fig.savefig(os.path.join(fig_path_final, name+'.eps'), bbox_inches='tight')
			fig.savefig(os.path.join(fig_path_final, name+'.png'), bbox_inches='tight')
			if flag_fig_close is True:
				plt.close(fig)

# ~ def clear_all():
	# ~ """Clear all the variables from the workspace of the spyder
	# ~ """
	# ~ gl=globals().copy()
	# ~ for var in gl:
		# ~ if var[0]=='_': continue
		# ~ if 'func' in str(globals()[var]): continue
		# ~ if 'module' in str(globals()[var]): continue
		
		# ~ del globals()[var]

# ~ if __name__=="__main__":
	# ~ clear_all()
