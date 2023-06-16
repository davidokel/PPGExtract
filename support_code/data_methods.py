"""
data_methods.py

This module provides various methods for data processing and analysis.

Functions:
- dict_to_df(dictionary): Converts a dictionary into a pandas DataFrame.
- normalise_data(data, fs): Normalizes the given data using bandpass and lowpass filtering techniques.
- band_pass_filter(data, order, fs, low_cut, high_cut): Applies a bandpass filter to the given data.
- data_scaler(data): Scales the given data by adding a scaling factor to make it non-negative.
- get_signal_slopes(data, index1, index2): Calculates the slope of a line given two data points and their corresponding indices.
"""

import numpy as np
import scipy.signal as sp
import scipy.stats as stat
import pandas as pd
import matplotlib.pyplot as plt

def dict_to_df(dictionary):
    """
    Converts a dictionary into a pandas DataFrame.

    Args:
        dictionary (dict): The input dictionary to be converted.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the data from the input dictionary.

    Raises:
        ValueError: If the dictionary is empty or contains invalid data format.
    """
    if not dictionary:
        raise ValueError("The dictionary is empty.")

    keys = list(dictionary.keys())
    keys.reverse()

    df = pd.DataFrame()

    rows = []
    for key in dictionary.keys():
        row_data = dictionary[key]
        row_dict = {}
        try:
            for k in row_data.keys():
                if k == 'features':
                    row_dict.update(row_data.get(k, {}))
                else:
                    row_dict[k] = row_data.get(k)
        except AttributeError as e:
            raise ValueError(f"The dictionary contains invalid data format. "
                             f"Invalid attribute: '{k}'") from e
        rows.append(row_dict)

    df = pd.concat([df, pd.DataFrame(rows)])

    return df.reset_index(drop=True)

"""def normalise_data(data, fs):
    
    Normalizes the given data using bandpass and lowpass filtering techniques.

    Args:
        data (array-like): The input data to be normalized.
        fs (int): The sampling frequency of the data.

    Returns:
        array-like: The normalized data.

    Raises:
        ValueError: If an error occurs during filtering.
    
    sos_ac = sp.butter(2, [0.5, 12], btype='bandpass', analog=False, output='sos', fs=fs)
    sos_dc = sp.butter(4, (0.2 / (fs / 2)), btype='lowpass', analog=False, output='sos', fs=fs)
    try:
        ac = sp.sosfiltfilt(sos_ac, data, axis=-1, padtype='odd', padlen=None)
        dc = sp.sosfiltfilt(sos_dc, data, axis=-1, padtype='odd', padlen=None)
    except ValueError:
        min_padlen = int((len(data) - 1) // 2)
        ac = sp.sosfiltfilt(sos_ac, data, axis=-1, padtype='odd', padlen=min_padlen)
        dc = sp.sosfiltfilt(sos_dc, data, axis=-1, padtype='odd', padlen=min_padlen)

    normalised = 10 * (ac / dc)

    return normalised"""

def normalise_data(data, fs):
    min_val = np.min(data)
    max_val = np.max(data)
    normalised_data = (data - min_val) / (max_val - min_val)
    return normalised_data

def band_pass_filter(data, order, fs, low_cut, high_cut):
    """
    Applies a bandpass filter to the given data.

    Args:
        data (array-like): The input data to be filtered.
        order (int): The filter order.
        fs (int): The sampling frequency of the data.
        low_cut (float): The lower cutoff frequency of the bandpass filter.
        high_cut (float): The upper cutoff frequency of the bandpass filter.

    Returns:
        array-like: The filtered data.

    Raises:
        ValueError: If an error occurs during filtering.
    """
    try:
        sos = sp.butter(order, [low_cut, high_cut], btype='bandpass', analog=False, output='sos', fs=fs)
        filtered_data = sp.sosfiltfilt(sos, data, axis=-1, padtype='odd', padlen=None)
    except ValueError as e:
        raise ValueError("An error occurred during filtering. Check the filter parameters and data compatibility.") from e

    return filtered_data

def data_scaler(data):
    """
    Scales the given data by adding a scaling factor to make it non-negative.

    Args:
        data (array-like): The input data to be scaled.

    Returns:
        array-like: The scaled data.

    Raises:
        ValueError: If the data is empty or cannot be scaled.
    """
    if len(data) == 0:
        raise ValueError("The data is empty.")

    min_data = min(data)
    scaling_factor = 0

    if min_data < 0:
        scaling_factor = abs(min_data)
    else:
        scaling_factor = -min_data

    scaled_data = data + scaling_factor

    return scaled_data

def get_signal_slopes(data, index1, index2):
    """
    Calculates the slope of a line given two data points and their corresponding indices.

    Args:
        data (array-like): The input data.
        index1 (int): The index of the first data point.
        index2 (int): The index of the second data point.

    Returns:
        float: The slope of the line.

    Raises:
        ValueError: If the indices are out of range or index2 is equal to index1.
    """
    if index1 < 0 or index1 >= len(data) or index2 < 0 or index2 >= len(data):
        raise ValueError("Invalid indices. The indices should be within the range of the data.")

    if index1 == index2:
        raise ValueError("The indices should be different.")

    y_diff = data[index2] - data[index1]
    x_diff = index2 - index1

    slope = y_diff / x_diff

    return slope
