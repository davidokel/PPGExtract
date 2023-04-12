import matplotlib.pyplot as plt
from support_code.data_methods import normalise_data, get_signal_slopes
import numpy as np
import scipy.signal as sp

def get_pulses(data,fs=100,visualise=False,debug=False):
    normalised_data = normalise_data(data,fs)

    if debug:
        plt.subplot(2,1,1)
        plt.title("Raw Data")
        plt.plot(data)
        plt.subplot(2,1,2)
        plt.title("Normalised Data")
        plt.plot(normalised_data)
        plt.show()
    
    # Filter data using savgol filter
    data_savgol = sp.savgol_filter(normalised_data, 51, 3)

    if debug:
        # Plot a subplot with the normalised data and the filtered data
        plt.subplot(2,1,1)
        plt.title("Normalised Data")
        plt.plot(normalised_data)
        plt.subplot(2,1,2)
        plt.title("Filtered Data")
        plt.plot(data_savgol)
        plt.show()
        
    # Calculate the first and second derivative of the data
    # Using np.gradient() to calculate the first derivative and second derivative
    # Using np.gradient instead of np.diff as np.gradient returns the same number of elements as the original array
    data_first_derivative = np.gradient(data_savgol)
    data_second_derivative = np.gradient(data_first_derivative)

    # Compute the element-wise difference between the two arrays
    diff = data_first_derivative - data_second_derivative
    
    # Find the indices where the sign changes above 0.001
    sign_change = np.where(np.gradient(np.sign(diff)))[0]
    
    # Add one to the indices to account for the offset introduced by np.diff
    crossings = sign_change
    # crossings = sign_change + 1

    # With an fs of 100Hz calculate what 220bpm equates to in terms of samples
    """bpm_220 = int((60/220)*fs)

    # Remove any crossings that are less than 220bpm apart
    mask = crossings > bpm_220
    crossings = crossings[mask]"""

    # Finding the peaks from the crossings
    peaks = []
    troughs = []
    notches = []
    peak_points = {}

    ##############################
    # CLASSIFY PEAKS AND TROUGHS #
    ##############################
    # If there are no crossings then return an empty list
    if len(crossings) == 0:
        return peaks
    else:
        # Iterate over the crossing points starting from the second crossing and ending at the second to last crossing
        for i in range(1, len(crossings)-1):
            pre_data_slope = get_signal_slopes(data_savgol, crossings[i-1], crossings[i])
            post_data_slope = get_signal_slopes(data_savgol, crossings[i], crossings[i+1])

            # If the slope of the data before the crossing is positive and the slope of the data after the crossing is negative then the crossing is a peak
            if pre_data_slope > 0 and post_data_slope < 0:
                peaks.append(crossings[i])

            # If the slope of the data before the crossing is negative and the slope of the data after the crossing is positive then the crossing is a trough
            if pre_data_slope < 0 and post_data_slope > 0:
                troughs.append(crossings[i])

    """# Calculating the upper and lower envelope of the signal using scipy interp1d and the peaks and troughs
    peaks_interpolated = np.interp(np.arange(0, len(data_savgol)), peaks, data_savgol[peaks])
    troughs_interpolated = np.interp(np.arange(0, len(data_savgol)), troughs, data_savgol[troughs])

    # Calculate the upper and lower envelope
    upper_envelope = np.maximum(peaks_interpolated, troughs_interpolated)
    lower_envelope = np.minimum(peaks_interpolated, troughs_interpolated)
    """

    # Calculating by how much np.gradient offsets the data
    offset = int((len(data_savgol) - len(diff))/2)

    # Adding offset to the left hand side of the first derivative
    data_first_derivative = np.insert(data_first_derivative, 0, np.zeros(offset))

    if debug:
        plt.plot(data_savgol)
        plt.title("CLASSIFY PEAKS AND TROUGHS")
        plt.plot(peaks, data_savgol[peaks], "gx")
        plt.plot(troughs, data_savgol[troughs], "rx")
        # Plot the first and second derivative on the same plot
        plt.plot(data_first_derivative*5)
        plt.plot(data_second_derivative*5)
        plt.show()
        # Plot the upper and lower envelope
        """plt.plot(upper_envelope)
        plt.plot(lower_envelope)"""

    # Go through the peaks and troughs and find the associated local maxima and minima
    peaks_raw = []
    troughs_raw = []

    search_space = int(fs*0.35)

    # Iterate over the peaks and troughs
    for i in range(0, len(peaks)):
        # Define the exploration space for the peak
        if peaks[i] - search_space < 0:
            peak_explore_start = 0
        else:
            peak_explore_start = peaks[i] - search_space
            
        if peaks[i] + search_space > len(normalised_data):
            peak_explore_end = len(normalised_data)
        else:
            peak_explore_end = peaks[i] + search_space

        # Find the local maxima and minima
        peak_raw = np.argmax(normalised_data[peak_explore_start:peak_explore_end]) + peak_explore_start

        # Add the local maxima and minima to the list
        peaks_raw.append(peak_raw)

    for i in range(0, len(troughs)):
        # Define the exploration space for the trough
        if troughs[i] - search_space < 0:
            trough_explore_start = 0
        else:
            trough_explore_start = troughs[i] - search_space

        if troughs[i] + search_space > len(normalised_data):
            trough_explore_end = len(normalised_data)
        else:
            trough_explore_end = troughs[i] + search_space

        trough_raw = np.argmin(normalised_data[trough_explore_start:trough_explore_end]) + trough_explore_start
        
        troughs_raw.append(trough_raw)

    while (len(peaks_raw) > 0 and len(troughs_raw) > 0) and (peaks_raw[0] < troughs_raw[0]):
        # If the first detected element is a peak not a trough then remove it until the first detected element is a trough
        peaks_raw.pop(0)
        # Plot the data

    # If the last detected element is a peak not a trough then remove it until the last detected element is a trough
    while (len(peaks_raw) > 0 and len(troughs_raw) > 0) and (peaks_raw[-1] > troughs_raw[-1]):
        peaks_raw.pop(-1)

    # Iterate over the troughs and the peaks, if there are more than one peaks between two troughs then remove the peaks that are not the local maxima
    for i in range(0, len(troughs_raw)-1):
        # Get the peaks between the current trough and the next trough
        peaks_between_troughs = [peak for peak in peaks_raw if peak > troughs_raw[i] and peak < troughs_raw[i+1]]

        # If there are more than one peaks between the current trough and the next trough
        if len(peaks_between_troughs) > 1:
            # Get the values of the normalised_data at the peaks between the current trough and the next trough
            peaks_between_troughs_values = [normalised_data[peak] for peak in peaks_between_troughs]

            # Get the index of the local maxima
            local_maxima_index = np.argmax(peaks_between_troughs_values)

            # Remove the peaks that are not the local maxima
            for j, peak in enumerate(peaks_between_troughs):
                if j != local_maxima_index:
                    peaks_raw.remove(peak)

    if debug:
        plt.plot(normalised_data)
        plt.title("PRE-DICTIONARY")
        plt.plot(list(map(int, peaks_raw)), [normalised_data[i] for i in map(int, peaks_raw)], "gx")
        plt.plot(list(map(int, troughs_raw)), [normalised_data[i] for i in map(int, troughs_raw)], "rx")
        plt.show()

    # Iterate over the peaks
    if len(peaks_raw) != 0:
        # Enumerate peaks_raw
        for i, peak in enumerate(peaks_raw):
            # If the peak is the first and only peak in peaks_raw and peak_points is empty
            if i == 0 and len(peaks_raw) == 1 and len(peak_points) == 0:
                # Get all troughs that are before the peak and after the peak
                pre_peak_troughs = [trough for trough in troughs_raw if trough < peak]
                post_peak_troughs = [trough for trough in troughs_raw if trough > peak]

                # Check that there are pre and post peak troughs
                if len(pre_peak_troughs) != 0 and len(post_peak_troughs) != 0:
                    # Get the max peak from pre_peak_troughs and the min peak from post_peak_troughs
                    pre_peak = max(pre_peak_troughs)
                    post_peak = min(post_peak_troughs)

                    relative_peak = peak - pre_peak

                    peak_points[peak] = {"Peak": peak, "Pre_peak": pre_peak, "Post_peak": post_peak, "raw_pulse_data": data[pre_peak:post_peak], "norm_pulse_data": normalised_data[pre_peak:post_peak], "Relative_peak": relative_peak}
                else: 
                    continue

            # If the peak is the first peak but not the last peak in peaks_raw and peak_points is empty
            elif i == 0 and len(peaks_raw) != 1 and len(peak_points) == 0:
                # Get all troughs that are before the peak and after the peak
                pre_peak_troughs = [trough for trough in troughs_raw if trough < peak]
                post_peak_troughs = [trough for trough in troughs_raw if trough > peak]

                # Check that there are pre and post peak troughs
                if len(pre_peak_troughs) != 0 and len(post_peak_troughs) != 0:
                    # Get the max peak from pre_peak_troughs and the min peak from post_peak_troughs
                    pre_peak = max(pre_peak_troughs)
                    post_peak = min(post_peak_troughs)

                    relative_peak = peak - pre_peak

                    if post_peak < peaks_raw[i+1]:
                        peak_points[peak] = {"Peak": peak, "Pre_peak": pre_peak, "Post_peak": post_peak, "raw_pulse_data": data[pre_peak:post_peak], "norm_pulse_data": normalised_data[pre_peak:post_peak], "Relative_peak": relative_peak}
                else: 
                    continue

            # If the peak is not the first nor last peak in peaks_raw and peak_points is not empty
            elif i != 0 and i != len(peaks_raw)-1 and len(peak_points) != 0:
                # Get all troughs that are before the peak and after the peak
                pre_peak_troughs = [trough for trough in troughs_raw if trough < peak]
                post_peak_troughs = [trough for trough in troughs_raw if trough > peak]

                # Check that there are pre and post peak troughs
                if len(pre_peak_troughs) != 0 and len(post_peak_troughs) != 0:
                    # Get the max peak from pre_peak_troughs and the min peak from post_peak_troughs
                    pre_peak = max(pre_peak_troughs)
                    post_peak = min(post_peak_troughs)

                    # Get peak_points keys
                    peak_points_keys = list(peak_points.keys())
                    # Get the last key in peak_points_keys
                    last_key = peak_points_keys[-1]

                    relative_peak = peak - pre_peak
                    
                    # Ensure that the pre_peak is >= the previous peak's post_peak and the post_peak is not greater than the next peak
                    if pre_peak >= peak_points[last_key]["Post_peak"] and post_peak < peaks_raw[i+1]:
                        # Add the peaks and troughs to the dictionary
                        peak_points[peak] = {"Peak": peak, "Pre_peak": pre_peak, "Post_peak": post_peak, "raw_pulse_data": data[pre_peak:post_peak], "norm_pulse_data": normalised_data[pre_peak:post_peak], "Relative_peak": relative_peak}
                else:
                    continue

            # If the peak is the last peak in peaks_raw and peak_points is not empty
            elif i == len(peaks_raw)-1 and len(peak_points) != 0:
                # Get all troughs that are before the peak and after the peak
                pre_peak_troughs = [trough for trough in troughs_raw if trough < peak]
                post_peak_troughs = [trough for trough in troughs_raw if trough > peak]

                # Check that there are pre and post peak troughs
                if len(pre_peak_troughs) != 0 and len(post_peak_troughs) != 0:
                    # Get the max peak from pre_peak_troughs and the min peak from post_peak_troughs
                    pre_peak = max(pre_peak_troughs)
                    post_peak = min(post_peak_troughs)

                    # Get peak_points keys
                    peak_points_keys = list(peak_points.keys())
                    # Get the last key in peak_points_keys
                    last_key = peak_points_keys[-1]

                    relative_peak = peak - pre_peak
                    
                    # Ensure that the pre_peak is >= the previous peak's post_peak
                    if pre_peak >= peak_points[last_key]["Post_peak"]:
                        # Add the peaks and troughs to the dictionary
                        peak_points[peak] = {"Peak": peak, "Pre_peak": pre_peak, "Post_peak": post_peak, "raw_pulse_data": data[pre_peak:post_peak], "norm_pulse_data": normalised_data[pre_peak:post_peak], "Relative_peak": relative_peak}
                else:
                    continue

        # Get all the pre and post peak points from the dictionary and store them in a list
        troughs = []
        peaks = []
        for key in peak_points:
            troughs.append(peak_points[key]["Pre_peak"])
            troughs.append(peak_points[key]["Post_peak"])
            
            # Get the index of the maximum value between the pre_peak and post_peak points
            corrected_peak = np.argmax(normalised_data[peak_points[key]["Pre_peak"]:peak_points[key]["Post_peak"]]) + peak_points[key]["Pre_peak"]
            
            # Change the peak value in the dictionary to the maximum value between the pre_peak and post_peak
            peak_points[key]["Peak"] = corrected_peak

            peaks.append(peak_points[key]["Peak"])
            
        # Plot the data and the peaks and troughs
        if debug or visualise:
            plt.plot(normalised_data)
            plt.plot(list(map(int, peaks)), [normalised_data[i] for i in map(int, peaks)], "go")
            plt.plot(list(map(int, troughs)), [normalised_data[i] for i in map(int, troughs)], "ro")
            plt.show()

    return peak_points, peaks, troughs