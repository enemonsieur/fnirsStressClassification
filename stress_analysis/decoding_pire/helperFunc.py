"""
This function contain method and classes which don't belong to any other categories
"""
import os
import re
from pathlib import Path, PureWindowsPath
from typing import List
import scipy.io as scio
import numpy as np
import mne
from mne import io


def get_raw_path(root_path, join_path):

	"""
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
	raw_data_folder_path=Path(root_path).joinpath(*join_path)
	return raw_data_folder_path


