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
    data_savgol = sp.savgol_filter(normalised_data, 74, 5)

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
    d1_norm_data = np.gradient(normalised_data)
    d2_norm_data = np.gradient(d1_norm_data)

    # Filter d1 and d2 using savgol filter
    d1_norm_data_filtered = sp.savgol_filter(d1_norm_data, 37, 5)
    # Median of the d1_norm_data_filtered
    d2_norm_data_filtered_median = np.median(d2_norm_data)
    d2_norm_data_filtered = sp.savgol_filter(d2_norm_data, 37, 5)

    # Compute the element-wise difference between the two arrays
    diff_norm_data = d1_norm_data_filtered - d2_norm_data_filtered
    
    # Find the indices where the sign changes above 0.001
    crossings_norm_data = np.where(np.gradient(np.sign(diff_norm_data)))[0]

    diff_data = normalised_data - np.mean(normalised_data)
    crossings_data = np.where(np.gradient(np.sign(diff_data)))[0]

    d1_savgol_data = np.gradient(data_savgol)
    d2_savgol_data = np.gradient(d1_savgol_data)

    # Compute the element-wise difference between the two arrays
    diff_savgol_data = d1_savgol_data - d2_savgol_data
    
    # Find the indices where the sign changes
    crossings_savgol_data = np.where(np.gradient(np.sign(diff_savgol_data)))[0]

    # Plot the data and the crossings
    if debug:
        plt.plot(data_savgol)
        plt.title("CROSSINGS")
        plt.plot(crossings_savgol_data, data_savgol[crossings_savgol_data], "gx")
        plt.show()

    # Finding the peaks from the crossings
    peaks = []
    troughs = []
    notches = []
    peak_points = {}

    ##############################
    # CLASSIFY PEAKS AND TROUGHS #
    ##############################
    # If there are no crossings then return an empty list
    if len(crossings_savgol_data) == 0:
        return peaks
    else:
        # Iterate over the crossing points starting from the second crossing and ending at the second to last crossing
        for i in range(1, len(crossings_savgol_data)-1):
            pre_data_slope = get_signal_slopes(data_savgol, crossings_savgol_data[i-1], crossings_savgol_data[i])
            post_data_slope = get_signal_slopes(data_savgol, crossings_savgol_data[i], crossings_savgol_data[i+1])

            # If the slope of the data before the crossing is positive and the slope of the data after the crossing is negative then the crossing is a peak
            if pre_data_slope > 0 and post_data_slope < 0:
                peaks.append(crossings_savgol_data[i])

            # If the slope of the data before the crossing is negative and the slope of the data after the crossing is positive then the crossing is a trough
            if pre_data_slope < 0 and post_data_slope > 0:
                troughs.append(crossings_savgol_data[i])

    while (len(peaks) > 0 and len(troughs) > 0) and (peaks[0] < troughs[0]):
        # If the first detected element is a peak not a trough then remove it until the first detected element is a trough
        peaks.pop(0)
        # Plot the data

    # If the last detected element is a peak not a trough then remove it until the last detected element is a trough
    while (len(peaks) > 0 and len(troughs) > 0) and (peaks[-1] > troughs[-1]):
        peaks.pop(-1)
    
    if debug:
        plt.plot(data_savgol)
        plt.title("CLASSIFY PEAKS AND TROUGHS")
        plt.plot(peaks, data_savgol[peaks], "gx")
        plt.plot(troughs, data_savgol[troughs], "rx")
        # Plot the first and second derivative on the same plot
        plt.plot(d1_savgol_data*5)
        plt.plot(d2_savgol_data*5)
        # Plot the third derivative on the same plot
        plt.plot(np.gradient(d2_savgol_data)*5)

        plt.show()
    
    candidate_peaks = []
    search_indexes = []
    pulse_centroids = []
    crossing_points = []

    # Iterate over the peaks
    for i in range(0, len(peaks)):
        # For the current peak, find the crossings_data which is closest to the left of the peak
        all_crossings_pre = crossings_data[crossings_data < peaks[i]]
        # Check that there are crossings before the peak
        if len(all_crossings_pre) != 0:
            left_crossing = max(all_crossings_pre)
            crossing_points.append(left_crossing)
        else:
            continue
    
        # For the current peak, find the crossings_data which is closest to the right of the peak
        all_crossings_post = crossings_data[crossings_data > peaks[i]]
        # Check that there are crossings after the peak
        if len(all_crossings_post) != 0:
            right_crossing = min(all_crossings_post)
            crossing_points.append(right_crossing)
        else:
            continue
        
        # If left_crossing and right_crossing exist:
        # Find all the crossings in crossings_norm_data that are between the left_crossing_norm and right_crossing_norm or not NaN
        if left_crossing != None and right_crossing != None and (np.isnan(left_crossing) == False and np.isnan(right_crossing) == False):
            crossings_norm_data_between = [crossing for crossing in crossings_norm_data if crossing > left_crossing and crossing < right_crossing]
            # If the corrsings_norm_data_between is empty
            if len(crossings_norm_data_between) == 0:
                # Define the peak_centroid as the middle between the left_crossing and the right_crossing
                pulse_centroid = int(np.mean([left_crossing, right_crossing]))
            else:
                # Find the average value crossing_norm_data_between
                pulse_centroid = int(np.mean(crossings_norm_data_between))
            
            pulse_centroids.append(pulse_centroid)

            # Calculate the distance between the left_crossing and the peak_centroid and the right_crossing and the peak_centroid
            left_distance = pulse_centroid - left_crossing
            right_distance = right_crossing - pulse_centroid

            # Calculate the average distance between the left and right distance
            average_distance = int(np.mean([left_distance, right_distance]))

            # Define the search space from peak_centroid - average_distance to peak_centroid + average_distance
            """search_start = int(pulse_centroid - average_distance)
            search_end = int(pulse_centroid + average_distance)"""

            search_start = int(left_crossing)
            search_end = int(right_crossing)

            search_indexes.append(search_start)
            search_indexes.append(search_end)
            
            # Find the index of the max value of the normalised_data between the search_start and search_end
            candidate_peak = np.argmax(normalised_data[search_start:search_end]) + search_start

            # Add the candidate peak to the list
            candidate_peaks.append(candidate_peak)
        else:
            continue

    if debug == True:
        plt.plot(normalised_data)
        plt.plot(data_savgol)
        # Plot the current peak
        plt.plot(peaks, data_savgol[peaks], "gx")
        # Plot the candidate peak as a green dot on the norm data
        plt.plot(candidate_peaks, normalised_data[candidate_peaks], "go")
        # Plotting crossings_data as yellow dots
        plt.plot(crossings_norm_data, d1_norm_data_filtered[crossings_norm_data], "yo")
        # Plot the crossing points as red dots on the norm data
        plt.plot(crossing_points, normalised_data[crossing_points], "ro")
        # Plot the pulse_centroids as blue dots
        plt.plot(pulse_centroids, d1_norm_data_filtered[pulse_centroids], "bo")
        # Plot the search_indexes list as vertical dashed black lines
        for i in range(0, len(search_indexes)):
            plt.axvline(x=search_indexes[i], color="k", linestyle="--")
        # Plot the d2_norm_data_filtered_median as a black horizontal line
        plt.axhline(y=np.mean(normalised_data), color="k")
        # Plot the d1_norm_data_filtered
        plt.plot(d1_norm_data_filtered)
        # Maximise the plot using figure manager
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.show()

        plt.plot(normalised_data)
        # Plot the candidate peaks
        plt.plot(candidate_peaks, normalised_data[candidate_peaks], "go")
        plt.show()

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

    if debug: 
        plt.plot(normalised_data)
        plt.plot(d1_norm_data_filtered)
        # Plot the d1_norm_data_filtered_median as a black horizontal line
        plt.axhline(y=d2_norm_data_filtered_median, color="k")
        plt.plot(data_savgol)

        # Plot current peak from peaks in data_savgol
        plt.plot(peaks, data_savgol[peaks], "gx")
        # Plot current peak in normalised_data
        plt.plot(peaks_raw, normalised_data[peaks_raw], "go")
        
        # Plot current trough from troughs in data_savgol
        plt.plot(troughs, data_savgol[troughs], "rx")
        # Plot current peak in normalised_data
        plt.plot(troughs_raw, normalised_data[troughs_raw], "ro")

        # Plot the d1_d2_cross as red dots
        plt.plot(crossings_norm_data, d1_norm_data_filtered[crossings_norm_data], "ko")
        plt.show()
        

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