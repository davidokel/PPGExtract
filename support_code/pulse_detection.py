import matplotlib.pyplot as plt
from support_code.data_methods import normalise_data, get_signal_slopes
import numpy as np
import scipy.signal as sp
from collections import Counter

def get_pulses(data, fs=100, visualise=False, debug=False):
    # Calculate a moving average of the data
    # Using np.convolve() to calculate the moving average
    # Using np.convolve instead of np.mean as np.convolve returns the same number of elements as the original array
    # Using a 1 second window
    # Flip the data by multiplying by -1 to get the peaks instead of the troughs
    window = int(fs*0.45)
    data = np.array(data)
    data = data * -1
    data = sp.savgol_filter(data, 7, 5)

    sos_ac = sp.butter(2, 4, btype='lowpass', analog=False, output='sos', fs=fs)
    try:
        data = sp.sosfiltfilt(sos_ac, data, axis= -1, padtype='odd', padlen=None)
    except ValueError:
        min_padlen = int((len(data) - 1) // 2)
        data = sp.sosfiltfilt(sos_ac, data, axis= -1, padtype='odd', padlen=min_padlen)

    moving_average_data = np.convolve(data, np.ones((window,))/window, mode='valid')
    # Add the missing elements to the moving average data
    # Get the difference between the length of the moving average data and the length of the original data
    # Remove n/2 elements from the start and end of the data
    # Add n/2 elements to the start and end of the moving average data
    # This is done to ensure that the moving average data is the same length as the original data
    n = len(data) - len(moving_average_data)
    data = data[int(n/2):-int(n/2)]

    # Find indexes where the moving average and the data intersect
    diff = moving_average_data - data
    crossings = np.where(np.gradient(np.sign(diff)))[0]

    # Plot the raw data and the moving average
    if visualise:
        plt.plot(data, label='Raw Data')
        plt.plot(moving_average_data, label='Moving Average')
        plt.plot(crossings, moving_average_data[crossings], 'ro', label='Crossings')
        plt.legend()
        plt.show()