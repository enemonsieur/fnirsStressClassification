# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

import numpy as np
import mne
import os, sys

from warnings import warn
import scipy.stats as sps
from nilearn._utils.glm import z_score
DEF_TINY = 1e-50
DEF_DOFMAX = 1e10

import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import mne
from mne.utils import warn
from mne.channels.layout import _merge_ch_data
from mne.io.pick import _picks_to_idx




def create_grouping_optodes(SD_list, oxy_type='hbo'):
	
	ch_list=[]
	for lst in SD_list:
		lst=[tuple(map(str, el)) for el in lst]
		src, det=zip(*lst)
		src, det=list(src),list(det)
		src=['S{0}_'.format(i) for i in src]
		det=[('D{0} '+oxy_type).format(i) for i in det]
		ch_tup=list(zip(src, det))
		ch_f=[''.join(tup) for tup in ch_tup]
		ch_list.append(ch_f)
	
	return ch_list
	
def get_row_avg_feat_idx(data, ch_idx_list):
    '''
    Take the average of the columns mentioned in the ch_idx_list and return it

    Parameters
    ----------
    data : 2D Array (row: data points; col: channels)
        Data to be averaged where each row represents data points and columns represent the channels
    ch_idx_list : List variable
        Holds the channel index that need to combine

    Returns
    -------
    avg_data : Arrau
        Channel averaged data points (row: same as data; col: reduce to len(ch_idx_list))

    '''
    # ~ [r, c]=data.shape
   	# ~ avg_data=np.array(data.shape[0], len(ch_idx_list))
   	# ~ flag=1
    avg_data=[]
    for ch_list in ch_idx_list:
   		row_means=data[:, ch_list].mean(axis=1)	
   		avg_data.append(row_means)
    
    avg_data=np.asarray(avg_data).T
    return avg_data



def pearsonr_ci(x, y, alpha=0.05):
	''' calculate Pearson correlation along with the confidence interval using scipy and numpy
	Parameters
	----------
	x, y : iterable object such as a list or np.array
	  Input for correlation calculation
	alpha : float
	  Significance level. 0.05 by default
	Returns
	-------
	r : float
	  Pearson's correlation coefficient
	pval : float
	  The corresponding p value
	lo, hi : float
	  The lower and upper bound of confidence intervals
	'''
	from scipy import stats

	r, p = stats.pearsonr(x,y)
	r_z = np.arctanh(r)
	se = 1/np.sqrt(len(x)-3)
	z = stats.norm.ppf(1-alpha/2)
	lo_z, hi_z = r_z-z*se, r_z+z*se
	lo, hi = np.tanh((lo_z, hi_z))
	return r, r*r, p, lo, hi

def predict_label(clf, X_scaled, y, n_splits):
	"""
	
	"""
	X_folds=np.array_split(X_scaled, n_splits) # make many folds of the data specified by n_splits
	y_folds=np.array_split(y, n_splits) # make many folds of the data specified by n_splits

	predict_y=[]
	for k in range(n_splits):
		X_train=list(X_folds) # convert into list (although it was list)
		X_test=X_train.pop(k) # take the first list for testing, rest(k-1 folds) for training 
		X_train=np.concatenate(X_train) # combine k-1 folds into array for training
		
		y_train=list(y_folds) # convert into list (although it was list)
		y_test=y_train.pop(k) # take the first list for testing, rest(k-1 folds) for training 
		y_train=np.concatenate(y_train) # combine k-1 folds into array for training
		
		clf.fit(X_train, y_train)
		temp_y=clf.predict(X_test)
		predict_y.extend(temp_y)
		del temp_y
		
	return np.array(predict_y)




## Read excel file from a specified path

def read_excel_sheets(xls_path, sheet=None):
	# If sheet is not specified, load the first sheet. Otherwise, 
	# load the specified sheet
	import pandas as pd
	if sheet is None:
		df=pd.read_excel(xls_path, engine='openpyxl')	
	else:
		df=pd.read_excel(xls_path, sheet_name=sheet, engine='openpyxl')
	return df

