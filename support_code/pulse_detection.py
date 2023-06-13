import matplotlib.pyplot as plt
from support_code.data_methods import normalise_data
import numpy as np
import scipy.signal as sp
import pandas as pd

def filter_troughs(p_t_tuples, data):
    """
    Filters troughs within the given p_t_tuples list based on the data.

    Args:
        p_t_tuples (list): List of tuples containing peak-trough information.
        data (array-like): The input data used for filtering troughs.

    Returns:
        list: Filtered p_t_tuples list after removing troughs.

    Raises:
        ValueError: If the p_t_tuples list or data is empty.
        ValueError: If the format of tuples within p_t_tuples is incorrect.
    """
    if len(p_t_tuples) == 0:
        raise ValueError("The p_t_tuples list is empty.")

    if len(data) == 0:
        raise ValueError("The data is empty.")

    # Get all the peaks and troughs within the p_t_tuples list
    peaks = []
    troughs = []
    for index, t in p_t_tuples:
        if t == "Peak":
            peaks.append(index)
        elif t == "Trough":
            troughs.append(index)

    # Iterate over the peaks in overlapping pairs
    for i, j in zip(peaks, peaks[1:]):
        # Get the troughs between the two peaks
        troughs_between = [trough for trough in troughs if i < trough < j]

        # Check if there are more than one trough between the two peaks
        if len(troughs_between) > 1:
            # Calculate the gradient of data between the troughs
            gradient = np.gradient(data[min(troughs_between):max(troughs_between)+1])

            # Determine if the majority of the data between the troughs slopes upwards or downwards
            majority_slope = np.sum(gradient < 0) > len(gradient) / 2

            # Remove the trough with the higher value based on the majority slope direction
            troughs_to_remove = [k if majority_slope else l for k, l in zip(troughs_between, troughs_between[1:])]

            # Remove the troughs between the two peaks from the p_t_tuples list
            p_t_tuples = [(index, t) for (index, t) in p_t_tuples if not (index in troughs_to_remove and t == "Trough")]

    return p_t_tuples

def z_score(feature_data, threshold):
    """
    Computes the z-scores of feature_data and returns a boolean array indicating values below the threshold.

    Args:
        feature_data (array-like): Input feature data.
        threshold (float): The threshold value for z-scores. Defaults to 2.5.

    Returns:
        numpy.ndarray: Boolean array indicating values below the threshold.

    Raises:
        ValueError: If feature_data is empty or threshold is not a positive number.
    """

    if len(feature_data) == 0:
        raise ValueError("The feature_data is empty.")

    if threshold <= 0:
        raise ValueError("The threshold must be a positive number.")

    # Compute the mean and standard deviation of the feature_data
    mean, std = np.mean(feature_data), np.std(feature_data)

    # Compute the z-scores of the feature_data
    z_scores = np.abs((feature_data - mean) / std)

    # Return a boolean array indicating values below the threshold
    return z_scores < threshold

def get_pulses(data, fs=100, z_score_threshold = 2.75, visualise=False, debug=False, z_score_detection = False):
    """
    Detects pulses in the given data using peak and trough detection.

    Args:
        data (list): The input data as a list of numeric values.
        fs (int): The sampling frequency of the data.
        z_score_threshold (float): The threshold value for z-score anomaly detection.
        visualise (bool, optional): If True, displays a plot of the data with detected peaks and troughs. Default is False.
        debug (bool, optional): If True, displays a detailed plot for debugging purposes. Default is False.
        z_score_detection (bool, optional): If True, uses z-score anomaly detection to remove false peaks. Default is True.

    Returns:
        peak_points (dict): A dictionary containing the peak points data.
        peaks (list): A list of peak indexes.
        troughs (list): A list of trough indexes.

    Raises:
        ValueError: If the data is empty.
    """
    
    # Handling empty data
    if len(data) == 0:
        raise ValueError("The data is empty.")
    # Ensure the data is a list otherwise convert it to a list
    if not isinstance(data, list):
        data = list(data)
    # Handling if fs is not a positive number
    if fs <= 0:
        raise ValueError("The sampling frequency must be a positive number.")
    # Handling if z_score_threshold is not a positive number
    if z_score_threshold <= 0:
        raise ValueError("The z_score_threshold must be a positive number.")
    
    # Normalising the data
    normalised_data = normalise_data(data, fs)
    
    # Filtering the data using a 3Hz, 2nd order lowpass Butterworth filter
    sos_ac = sp.butter(2, 5, btype='lowpass', analog=False, output='sos', fs=fs)
    try:
        data = sp.sosfiltfilt(sos_ac, data, axis=-1, padtype='odd', padlen=None)
    except ValueError:
        min_padlen = int((len(data) - 1) // 2)
        data = sp.sosfiltfilt(sos_ac, data, axis=-1, padtype='odd', padlen=min_padlen)
        
    # Calculating the moving average of the data using a 0.95 second window
    moving_average_data = np.convolve(data, np.ones((int(fs * 0.95),)) / int(fs * 0.95), mode='valid')
    n = len(data) - len(moving_average_data)
    data = data[int(n/2):-int(n/2)]

    # Finding the crossing points of the data and the moving average
    diff = data - moving_average_data
    crossings = np.where(np.gradient(np.sign(diff)))[0]

    # Splitting the crossings into groups and combining groups that are close together (less than fs/10 samples apart)
    differences = np.diff(crossings)
    split_indices = np.where(differences > int(fs/10))[0] + 1
    groups = np.split(crossings, split_indices)
    crossings = [group[0] for group in groups if len(group) > 0]

    # Saving elements into the p_t_tuples list as (index, "Peak") or (index, "Trough")
    p_t_tuples = []

    ##############################################
    # DEFINING PEAKS AND TROUGHS WITHIN THE DATA #
    ##############################################
    # Iterating over the crossings in overlapping pairs
    for i, j in zip(crossings, crossings[1:]):
        # Calculate the average moving average between the crossings
        average_moving_average = np.mean(moving_average_data[i:j]) 
        
        # Get the data between the two crossings and calculate the mean of the data
        data_between = data[i:j]
        mean_data_between = np.mean(data_between)
        
        # Determine if the data between the two crossings is a peak or trough
        # If the mean of the data between the two crossings is greater than the average moving average, it is a peak
        if mean_data_between > average_moving_average:
            peak = np.argmax(data_between) + i # Get the argmax of the data between the two crossings and add i to get the index of the peak

            # Check if the previous element in p_t_tuples is a peak
            if len(p_t_tuples) > 0 and p_t_tuples[-1][1] == "Peak":
                # Get the value of the previous peak and the current peak
                previous_peak = p_t_tuples[-1][0]
                previous_peak_value = data[previous_peak]
                current_peak_value = data[peak]
                
                # If the current peak's value is greater than the previous peak's value, remove the previous peak from p_t_tuples
                if current_peak_value > previous_peak_value:
                    p_t_tuples.pop()

            # Add the current peak to p_t_tuples
            p_t_tuples.append((peak, "Peak"))
        
        # If the mean of the data between the two crossings is less than the average moving average, it is a trough
        elif mean_data_between < average_moving_average:
            trough = np.argmin(data_between) + i # Get the argmin of the data between the two crossings and add i to get the index of the trough
            p_t_tuples.append((trough, "Trough"))

    ####################################################
    # ENSURING THE FIRST AND LAST ELEMENTS ARE TROUGHS #
    ####################################################
    while len(p_t_tuples) > 0 and (p_t_tuples[0][1] == "Peak" or p_t_tuples[-1][1] == "Peak"):
        if p_t_tuples[0][1] == "Peak":
            p_t_tuples.pop(0)
        if p_t_tuples and p_t_tuples[-1][1] == "Peak":
            p_t_tuples.pop(-1)

    ########################################
    # REMOVING ANOMALOUS PEAKS AND TROUGHS #
    ########################################
    # Anomalous peaks are peaks that have a value lower than either of the troughs either side of it.
    anom_peak_indexes = []
    for i in range(1, len(p_t_tuples) - 1):
        if p_t_tuples[i][1] == "Peak" and (data[p_t_tuples[i][0]] < data[p_t_tuples[i-1][0]] or data[p_t_tuples[i][0]] < data[p_t_tuples[i+1][0]]):
            anom_peak_indexes.append(p_t_tuples[i][0])
    p_t_tuples = [(index, label) for index, label in p_t_tuples if index not in anom_peak_indexes]

    # Filter the troughs in p_t_tuples
    p_t_tuples = filter_troughs(p_t_tuples, data)

    # Extract peaks and troughs from p_t_tuples
    peaks = [x[0] for x in p_t_tuples if x[1] == "Peak"]
    troughs = [x[0] for x in p_t_tuples if x[1] == "Trough"]

    peak_points = {}
    peaks = list(set(peaks))  # Remove peak duplicates

    for i in range(len(peaks)):
        left_troughs = [trough for trough in troughs if trough < peaks[i]]
        right_troughs = [trough for trough in troughs if trough > peaks[i]]
        pulse_onset = left_troughs[-1]
        pulse_end = right_troughs[0]

        key = peaks[i]
        peak_points[key] = {
            "Peak": peaks[i],
            "Relative_peak": peaks[i] - pulse_onset,
            "raw_pulse_data": data[pulse_onset:pulse_end],
            "norm_pulse_data": normalised_data[pulse_onset:pulse_end],
            "Pre_peak": pulse_onset,
            "Post_peak": pulse_end
        }

    if z_score_detection:
        # Calculating quality metrics i) inter-peak distance, ii) onset-peak difference (y), iii) peak-end difference (y)
        peak_distances = np.gradient(peaks)
        onset_peak_y = [data[peaks[i]] - data[left_troughs[-1]] for i, left_troughs in enumerate([[trough for trough in troughs if trough < peak] for peak in peaks])]
        peak_end_y = [data[peaks[i]] - data[right_troughs[0]] for i, right_troughs in enumerate([[trough for trough in troughs if trough > peak] for peak in peaks])]

        # Calculate z-scores for quality metrics
        peak_distances_z_scores = z_score(peak_distances, threshold=z_score_threshold)
        onset_peak_y_z_scores = z_score(onset_peak_y, threshold=z_score_threshold)
        peak_end_y_z_scores = z_score(peak_end_y, threshold=z_score_threshold)

        # Get the indexes where the z-scores are False
        false_indexes = np.concatenate((np.where(peak_distances_z_scores == False)[0],
                                        np.where(onset_peak_y_z_scores == False)[0],
                                        np.where(peak_end_y_z_scores == False)[0]))

        # Get indexes of the false peaks
        false_peaks = [peaks[index] for index in false_indexes]

    if visualise:
        plt.title("Pulse Detection")
        plt.plot(data, label='Data')
        plt.plot(peaks, data[peaks], 'go', label='Peaks')
        plt.plot(troughs, data[troughs], 'ro', label='Troughs')
        if z_score_detection:
            plt.plot(false_peaks, data[false_peaks], 'yo', label='Detected "Anomalous Peaks"')
        plt.legend()
        plt.show()

    if debug:
        plt.title("Pulse Detection")
        plt.plot(data, label='Data')
        plt.plot(moving_average_data,'b--', label='Moving Average')
        plt.plot(crossings, moving_average_data[crossings], 'bo', label='Crossings')
        plt.plot(peaks, data[peaks], 'go', label='Peaks')
        plt.plot(troughs, data[troughs], 'ro', label='Troughs')
        if z_score_detection:
            plt.plot(false_peaks, data[false_peaks], 'yo', label='Detected "Anomalous Peaks"')
        plt.legend()
        plt.show()

    # Removing the false peaks from peak_points dictionary if z_score_detection is True
    if z_score_detection and len(false_indexes) > 0:
        for index in false_indexes:
            key = peaks[index]
            if key in peak_points:
                peak_points.pop(key)

    peaks = [peak_points[key]["Peak"] for key in peak_points]
    troughs = [peak_points[key]["Pre_peak"] for key in peak_points] + [peak_points[key]["Post_peak"] for key in peak_points]

    if visualise:
        plt.title("Pulse Detection")
        plt.plot(data, label='Data')
        plt.plot(peaks, data[peaks], 'go', label='Peaks')
        plt.plot(troughs, data[troughs], 'ro', label='Troughs')
        plt.legend()
        plt.show()

    return peak_points, peaks, troughs