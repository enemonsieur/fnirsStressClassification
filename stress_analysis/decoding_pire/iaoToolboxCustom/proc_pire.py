"""
This proc.py file contains methods and classes for processing and prepocessing data
"""

import mne
from typing import List
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


class FilterParams:
    """

    Class Object which contains properties for mne filter parameters

    Parameters
    ----------
    l_freq:
    h_freq:
    picks:
    filter_length:
    l_trans_bandwidth:
    h_trans_bandwidth:
    n_jobs:
    method:
    iir_params:
    phase:
    fir_window:
    fir_design:
    skip_by_annotation:
    pad:
    verbose:

    Example
    --------
    filterParams = FilterParameters(
        l_freq=0.5,
        h_freq=22
    )

    """

    def __init__(self, l_freq, h_freq, picks=None, filter_length='auto', l_trans_bandwidth='auto',
                 h_trans_bandwidth='auto', n_jobs=1, method='fir', iir_params=None, phase='zero',
                 fir_window='hamming', fir_design='firwin', skip_by_annotation=('edge', 'bad_acq_skip'),
                 pad='reflect_limited', verbose=None):

        # set l_freq and h_freq outside of a dict
        self.l_freq = l_freq
        self.h_freq = h_freq

        self.params = dict(
            picks=picks,
            filter_length=filter_length,
            l_trans_bandwidth=l_trans_bandwidth,
            h_trans_bandwidth=h_trans_bandwidth,
            n_jobs=n_jobs,
            method=method,
            iir_params=iir_params,
            phase=phase,
            fir_window=fir_window,
            fir_design=fir_design,
            skip_by_annotation=skip_by_annotation,
            pad=pad,
            verbose=verbose
        )




class fNIRSreadParams:
	"""
	Class containing parameters for mne.io.read_raw_nirx
	
	parameters
	----------
	fname:     Path to the NIRX data folder or header file.
	preload: Preload data into memory for data manipulation and faster indexing
	verbose:
	
	Return
	------
	raw: a raw object containing NIRX data
		
	"""
	def __init__(self, preload=False, verbose=None):
		self.params=dict(
			preload=preload,
			verbose=verbose
		)




def filter(raw, fp: FilterParams, filt_type):
    """Filters data set

    This method filters the raw data set. The filter method depends on
    the chosen filt_type. It could be either bandpass, bandstop, highpass
    or lowpass.

    Parameter
    --------
    raw: mne.raw
        mne data-structure object
    fp: FilterParameters
        FilterParameter class object
    filt_type: str
        could be either bandpass, bandstop, highpass or lowpass

    Returns
    -------
    data: mne.raw
        filtered mne.raw data

    Raises
    ------
    ValueError
        If filt_type is not either bandpass, bandstop, highpass or
        lowpass

    Example
    -------
    filterParams = FilterParameters(
        l_freq=0.5,
        h_freq=22
    )
    data = proc.filter(mne_data, filterParams, 'bandpass')

    """

    types = ['bandpass', 'bandstop', 'highpass', 'lowpass']

    # Check if filt_type is among types
    if filt_type not in types:
        raise ValueError('filt_type is not defined')

    data = raw.copy()

    if filt_type == 'bandpass':
        data = raw.filter(l_freq=fp.l_freq, h_freq=fp.h_freq, **fp.params)

    if filt_type == 'bandstop':
        data = raw.filter(l_freq=fp.h_freq, h_freq=fp.l_freq, **fp.params)

    if filt_type == 'highpass':
        data = raw.filter(l_freq=fp.l_freq, h_freq=None, **fp.params)

    if filt_type == 'lowpass':
        data = raw.filter(l_freq=None, h_freq=fp.h_freq, **fp.params)

    return data



def load_fNIRS_data(path, info: fNIRSreadParams):
	"""
	Load fNIRS data"
	parameters
	----------
	path: path to the folder of fNIRS data or .hdr file
	info: fNIRSreadParams
	custom fNIRSreadParams class
	Returns
	-------
	data: mne.raw
		mne-fNIRS data-structure	
	"""
	raw_intensity=mne.io.read_raw_nirx(path, **info.params)
	
	return raw_intensity.load_data()

