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
    window = int(fs*0.95)
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
    if len(moving_average_data) >= len(data):
        diff = moving_average_data - data
    elif len(moving_average_data) <= len(data):
        diff = data - moving_average_data 

    crossings = np.where(np.gradient(np.sign(diff)))[0]

    # Find consecutive runs of indices
    differences = np.diff(crossings)
    split_indices = np.where(differences > 10)[0] + 1
    # Split the indices into separate groups
    groups = np.split(crossings, split_indices)
    # Summarize each group by taking the first index
    crossings = [group[0] for group in groups]

    troughs, peaks = [], []
    # Iterate over the crossings in pairs and check if the data between the crossings is above the moving average or not
    # If the data is above the moving average then the get the argmax of the data between the crossings
    # If the data is below the moving average then the get the argmin of the data between the crossings
    for i, j in zip(crossings, crossings[1:]):
        peak_or_trough = "Either"
        # Calculate the average moving average value between the two crossings
        average_moving_average = np.mean(moving_average_data[i:j])
        # Determine if the data between the two crossings is above or below the moving average
        if np.mean(data[i:j]) > average_moving_average:
            # Get the argmin of the data between the two crossings
            peak = np.argmax(data[i:j]) + i
            peaks.append(peak)
            peak_or_trough = "Peak"
        elif np.mean(data[i:j]) < average_moving_average:
            # Get the argmax of the data between the two crossings
            trough = np.argmin(data[i:j]) + i
            troughs.append(trough)
            peak_or_trough = "Trough"
        
        # Plot the raw data and the moving average
        if debug:
            plt.plot(data, label='Raw Data')
            plt.plot(moving_average_data, label='Moving Average')
            # Plot the current crossings
            plt.plot(i, moving_average_data[i], 'bo', label='Crossings')
            plt.plot(j, moving_average_data[j], 'bo', label='Crossings')
            # If the peak_or_trough is a peak then plot the peak
            if peak_or_trough == "Peak":
                plt.plot(peak, data[peak], 'go', label='Peak')
            # If the peak_or_trough is a trough then plot the trough
            elif peak_or_trough == "Trough":
                plt.plot(trough, data[trough], 'ro', label='Trough')
            plt.legend()
            plt.show()

    # Plot the raw data and the moving average
    if visualise:
        plt.plot(data, label='Raw Data')
        plt.plot(moving_average_data, label='Moving Average')
        plt.plot(crossings, moving_average_data[crossings], 'bo', label='Crossings')
        plt.plot(troughs, data[troughs], 'ro', label='Troughs')
        plt.plot(peaks, data[peaks], 'go', label='Peaks')
        plt.legend()
        plt.show()