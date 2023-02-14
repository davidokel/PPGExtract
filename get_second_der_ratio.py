import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp
import data_methods
from scipy.stats import linregress
from scipy.integrate import trapz

def get_second_der_ratio(data,fs):
    data = data.dropna().to_numpy()        
    # Normalise the distal_data and proximal_data using the normalise_data
    data = data_methods.normalise_data(data, 100)

    # Calling the get_peaks function from data_methods.py to find the peaks in the data
    peaks = data_methods.get_peaks(data, fs) # Given the data and the sampling frequency, get the peak locations

    second_derivative_ratio = 0

    if len(peaks) != 0:
        second_derivative = np.diff(np.diff(data))
        max_value = max(second_derivative)
        min_value = min(second_derivative)
        second_derivative_ratio = max_value/min_value

        return float(abs(second_derivative_ratio))
    else:
        return np.NaN