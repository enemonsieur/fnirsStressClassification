#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 14:27:17 2022

@author: pdhara
"""
# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def flatten_list(t):
    return [item for sublist in t for item in sublist]

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


def s1_func(k, i, x_i, T):
    '''
    The line "x_i-T[i-k:i]" comutes the difference between x_i and T[i-k:i] (i.e., k data points before x_i=T[i]).
    The line "x_i-T[i+1:i+k+1]" comutes the difference between x_i and T[i+1:i+k+1] (i.e., k data points after x_i=T[i]).
    Thereafter, takes the average of the maximum from the two above arrays.
    
    It is an implementation of S1 function in the following paper: Simple Algorithms for Peak Detection in Time-Series

    Parameters
    ----------
    k : TYPE
        Number of samples before or after i-th data point.
    i : TYPE
        Index of current data point from which k samples (before or after) will be subtracted
    x_i : TYPE
        Value at index i i.e., x_i=T[i]
    T : TYPE
        Time series data.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    N_minus_max=max(x_i-T[i-k:i])
    N_plus_max=max(x_i-T[i+1:i+k+1])
    return (N_minus_max+N_plus_max)/2



def m2d(x, i):
    '''
    This function return second derivative of x at point i

    Parameters
    ----------
    x : TYPE
        time series data
    i : TYPE
        Point at which second deivative will be computed.

    Returns
    -------
    TYPE
        Return second derivative of x at point i.

    '''
    return (-3*x[i-3]-2*x[i-2]-x[i-1]+x[i+1]+2*x[i+2]+3*x[i+3])/28

# def first_difference(x, i, T):
#     return (x[i]-x[i-1])/T

def first_difference_series(x, T):
    first_difference=np.zeros(len(x))
    for i in range(1, len(x)):
        first_difference[i]=(x[i]-x[i-1])/T
    return first_difference



def ibi_timeseries(pks, T, N):
    """
    This function generates time difference between two peaks (interbeat interval)
    
    Parameters
    ----------
    pks : array of integer
        contains peak index
    T : float
        Sampling duration
    N : TYPE
        Number of samples in the time series from which peak indexes are identified. 
        Here, N holds the number of samples in fnirs_1_data

    Returns
    -------
    ibi : array of float
        Returns time difference between two peaks (n1 and n2) at index n2. In between two peaks, 
        the indexes are filled with np.nan. 
    """
    ibi=np.empty(N) # creates an array of size of the original time series data from which peak indexes are derived
    ibi[:]=np.nan # assigned np.nan
    ibi[0], ibi[-1]=0, 0 # First and last sample value are set to zero
    for i in range(1, len(pks)):        
        ibi[pks[i]]=(pks[i]-pks[i-1])*T    
    return ibi

def ibi_interpolation(ibi, method='polynomial', order=2):
    """
    The computed interbeat interval (IBI) contain np.nan in between two peak indexes. Therefore, interpolation is done to 
    eliminate np.nan and generate a IBI series

    Parameters
    ----------
    ibi : array of float and np.nan
        DESCRIPTION.
    method : Interpolation method
        DESCRIPTION. The default is 'polynomial'.
    order : Order of polynomial interpolation
        DESCRIPTION. The default is 2.

    Returns
    -------
    array of float
        IBI time series of size the original time series.

    """
    ibi=pd.Series(ibi)
    ibi_inter=ibi.interpolate(method=method, order=2)
    return ibi_inter.values


def read_excel_sheets(xls_path, sheet=None):
	# If sheet is not specified, load the first sheet. Otherwise, 
	# load the specified sheet
	import pandas as pd
	if sheet is None:
		df=pd.read_excel(xls_path, engine='openpyxl')	
	else:
		df=pd.read_excel(xls_path, sheet_name=sheet, engine='openpyxl')
	return df