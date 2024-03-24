import numpy as np
import pandas as pd
from sklearn.utils import resample


def get_data(fnirs_data, d_type, t_tup):
	# return the data belong to the time interval
    #return fnirs_data.pick_types(fnirs=d_type)
	return fnirs_data.pick_types(fnirs=d_type).crop(tmin=t_tup[0], tmax=t_tup[1], include_tmax=True).get_data()

def get_diff_talking_base_data(raw, t, t_b=10, t_a=20):
	# This function takes the raw data, time instant as input. It will
	# collect the data 10 sec before and 20 sec after the specified time instant.
	# Then take the average of these data before and after this interval. Then compute
	# the difference of this two averages.
	
	base_data=get_data(raw.copy(), (t-t_b, t)).mean(axis=1)
	speaking_data=get_data(raw.copy(), (t, t+t_a)).mean(axis=1)
	diff=speaking_data-base_data
	return diff[:, np.newaxis]

def get_mismatched_ele(super_list, sub_list):
    '''
    This function takes two lists and return the elements that are present in super_list but not  in sub_list

    Parameters
    ----------
    super_list : TYPE
        super_list holds the superset of elements
    sub_list : TYPE
        sub-list that holds subset of elements present in super_list

    Returns
    -------
    missed_ele : list
        return the elements present in super_list but not in sub_list

    '''
   
    s=set(sub_list)
    missed_ele=[x for x in super_list if x not in s]
    return missed_ele


import numpy as np

from mne.viz import plot_alignment
from mne import verbose




def _open(fname):
    return open(fname, 'r', encoding='latin-1')

    

def get_time_avged_data(raw):
    '''
    Get average fNIRS data across channels. Let, there are 32 channels and 1000 data points.
	This function returns channel-averaged fNIRS data i.e., data of size 1x1000 

    Parameters
    ----------
    raw : fNIRS data object
        fNIRS data

    Returns
    -------
    times : array
        time information of the data
    data : 1-D array
        Averaged fNIRS data across channels

    '''
    data=raw.get_data().mean(axis=0)
    times=raw.times
    return times, data

def get_data(fnirs_data, d_type, t_tup):
 	# return the data belong to the time interval
 	return fnirs_data.pick_types(fnirs=d_type).crop(tmin=t_tup[0], tmax=t_tup[1], include_tmax=True).get_data()

# def get_diff_avg_epo_base(raw, t):
# 	# This function takes the raw data, time instant as input. It will
# 	# collect the data 10 sec before and 20 sec after the specified time instant.
# 	# Then take the average of these data before and after this interval. Then compute
# 	# the difference of this two averages.
# 	
# 	base_data=get_data(raw.copy(), (t[0]-2, t[0])).mean(axis=1)
# 	epoch_data=get_data(raw.copy(), (t[0], t[1])).mean(axis=1)
# 	diff=epoch_data-base_data
# 	return diff[:, np.newaxis]

## Balance the data
def get_balanced_data(df, col='Predict', scaling_type='downsample'):
    '''
    This function takes a dataframe of features and prediction value. It will calculate the 
    feature size for each classes, and balance the data based on user input.
    # https://elitedatascience.com/imbalanced-classes
    Parameters
    ----------
    df : dataframe
        Contains features (row: feature samples, columns: features)
    col : string
        The column refers to the prediction variable in the dataframe (such as 0, 1, etc) 
        DESCRIPTION. The default is 'Predict'.
    scaling_type : string: 'downsample' or upsample
        DESCRIPTION. The default is 'downsample'.
        based on this string, the min category would upsampled to the size of max category
        or reverse

    Returns
    -------
    df_final : dataframe
        After upscaling or downscaling the data, combine two dataframes into single dataframe.

    '''

    pr_class=np.unique(df[col]) # what are the unique predicting value (1, 0, -1, etc)
    if len(pr_class)>2: # If more than two class, just return to the main function
        print('Works for two class problem')
        return df
    pr_count=df[col].value_counts().to_numpy() # No. of appearance in each predicted class as an array from the dataframe
    if pr_count[0]==pr_count[1]: # if two classes have equal number of occurrences
        print('Each class has equal number of trials. No need to balance the data')
        return df
    pr_cla=df[col].value_counts().index    
    max_index=np.where(pr_count==np.max(pr_count))[0][0]
    min_index=np.where(pr_count==np.min(pr_count))[0][0]
    
    df_maj=df[df[col]==pr_cla[max_index]]
    df_min=df[df[col]==pr_cla[min_index]]
    if scaling_type=='downsample':
        df_maj_down=resample(df_maj, replace=False,
                         n_samples=pr_count[min_index],
                         random_state=123)
        
        df_final=pd.concat([df_min, df_maj_down])
    elif scaling_type=='upsample':
        df_min_up=resample(df_min, replace=True,
                           n_samples=pr_count[max_index],
                         random_state=123)
        df_final=pd.concat([df_min_up, df_maj])
        
    return df_final