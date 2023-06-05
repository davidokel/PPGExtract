import matplotlib.pyplot as plt
from support_code.data_methods import normalise_data, get_signal_slopes
import numpy as np
import scipy.signal as sp
from operator import itemgetter

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
    d1_norm_data_filtered = sp.savgol_filter(d1_norm_data, 47, 5)
    d2_norm_data_filtered = sp.savgol_filter(d2_norm_data, 47, 5)

    # Median of the d1_norm_data_filtered
    d2_norm_data_filtered_median = np.median(d2_norm_data)
    
    # CALCULATING THE CROSSING POINTS (FIRST AND SECOND DERIVATIVE)
    diff_norm_data = d1_norm_data_filtered - d2_norm_data_filtered
    crossings_norm_data = np.where(np.gradient(np.sign(diff_norm_data)))[0]

    # CALCULATING THE CROSSING POINTS (NORMALISED RAW DATA)
    diff_data = normalised_data - np.mean(normalised_data)
    crossings_data = np.where(np.gradient(np.sign(diff_data)))[0]

    # CALCULATING THE CROSSING POINTS (FIRST AND SECOND DERIVATIVE OF THE FILTERED DATA)
    d1_savgol_data = np.gradient(data_savgol)
    d2_savgol_data = np.gradient(d1_savgol_data)

    diff_savgol_data = d1_savgol_data - d2_savgol_data
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

    # Generate a dictionary called peak_points for storing peak information
    peak_points = {}
    search_indexes = []

    # Iterate over the peaks
    for i in range(0, len(peaks)):
        # For the current peak, find the crossings_data which is closest to the left of the peak
        all_crossings_pre = crossings_data[crossings_data < peaks[i]]
        # Check that there are crossings before the peak
        if len(all_crossings_pre) != 0:
            left_crossing = max(all_crossings_pre)
            # Add for the current peak and the left_crossing to the dictionary
            peak_points[i] = {"Peak": peaks[i], "Pre_peak": left_crossing}
        else:
            # Add a NaN value for the Pre_peak
            peak_points[i] = {"Peak": peaks[i], "Pre_peak": None}
    
        # For the current peak, find the crossings_data which is closest to the right of the peak
        all_crossings_post = crossings_data[crossings_data > peaks[i]]
        # Check that there are crossings after the peak
        if len(all_crossings_post) != 0:
            right_crossing = min(all_crossings_post)
            # Add for the current peak and the right_crossing to the dictionary
            peak_points[i]["Post_peak"] = right_crossing
        else:
            peak_points.pop(i)

    # Defining the pulse onset and end, in order to find the most accurate peaks
    # Enumerate the peak_points dictionary and get the keys and values
    # Get all keys in the peak_points dictionary
    peak_points_keys = list(peak_points.keys())
    # Iterate over the peak_points_keys
    for i, key in enumerate(peak_points_keys):
        print("Processing key {} of {}".format(i+1, len(peak_points)))
        # Print key values
        print(peak_points[key])
        # For the current peak, get the left_crossing and right_crossing
        left_crossing = peak_points[key]["Pre_peak"]
        right_crossing = peak_points[key]["Post_peak"]

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
            
            # Add the pulse_centriod to the peak_points dictionary
            peak_points[key]["Pulse_centroid"] = pulse_centroid

            # Get the closest crossings_norm_data which is less than left_crossing
            left_points = crossings_norm_data[crossings_norm_data < left_crossing]
            # Get the closest crossings_norm_data which is greater than right_crossing
            right_points = crossings_norm_data[crossings_norm_data > right_crossing]

            # If there are left_points and right_points
            if len(left_points) != 0 and len(right_points) != 0:
                pulse_onset_search = max(left_points)
                pulse_end_search = min(right_points)

                # Get distance between the left_crossing and the centroid
                distance_left_crossing_centroid = int(abs(pulse_centroid - left_crossing))
                # Get distance between the right_crossing and the centroid
                distance_right_crossing_centroid = int(abs(pulse_centroid - right_crossing))

                pulse_onset_search_start = int(pulse_onset_search - distance_left_crossing_centroid)
                pulse_onset_search_end = left_crossing

                pulse_end_search_start = right_crossing
                pulse_end_search_end = int(pulse_end_search + distance_right_crossing_centroid)

                # Adding elements to search_indexes
                search_indexes.append(pulse_onset_search_start)
                search_indexes.append(pulse_onset_search_end)
                search_indexes.append(pulse_end_search_start)
                search_indexes.append(pulse_end_search_end)
            
                # Define the search space within the normalised_data
                onset_search_space = normalised_data[pulse_onset_search_start:pulse_onset_search_end]
                end_search_space = normalised_data[pulse_end_search_start:pulse_end_search_end]

                # Check that the onset_search_space and end_search_space are not empty
                if len(onset_search_space) != 0 and len(end_search_space) != 0:
                    pulse_onset = np.argmin(onset_search_space) + pulse_onset_search_start
                    pulse_end = np.argmin(end_search_space) + pulse_end_search_start
                else:
                    continue

                # Get the index of the maximum value between the pulse_onset and pulse_end
                pulse_peak = np.argmax(normalised_data[pulse_onset:pulse_end]) + pulse_onset

                # Update the peak_points dictionary with the pulse_onset, pulse_end and pulse_peak
                peak_points[key]["Pre_peak"] = pulse_onset
                peak_points[key]["Post_peak"] = pulse_end
                peak_points[key]["Pulse_peak"] = pulse_peak

                # Calculate the peak prominence
                prominence = sp.peak_prominences(normalised_data, [pulse_peak])[0]
                # Add the prominence to the dictionary
                peak_points[key]["Prominence"] = prominence
        else:
            continue

    # Get all the pre and post peak points from the dictionary
    # Get a list of all "Pre_peak" and "Post_peak" keys from the dictionary using itemgetter
    pre_peaks = list(map(itemgetter("Pre_peak"), peak_points.values()))
    post_peaks = list(map(itemgetter("Post_peak"), peak_points.values()))
    # Get a list of all "Pulse_peak" keys from the dictionary using itemgetter
    pulse_peak_keys = list(map(itemgetter("Pulse_peak"), peak_points.values()))
    # Get a list of all "Pulse_centroid" keys from the dictionary using itemgetter
    pulse_centroid_keys = list(map(itemgetter("Pulse_centroid"), peak_points.values()))
    # Get a list of all "Prominence" keys from the dictionary using itemgetter
    prominences = list(map(itemgetter("Prominence"), peak_points.values()))
    # Combine the pre and post peaks list
    pre_post_peaks = pre_peaks + post_peaks

    # Calculate the upper and lower envelopes
    upper_envelope = np.interp(np.arange(len(normalised_data)), pulse_peak_keys, normalised_data[pulse_peak_keys])
    lower_envelope = np.interp(np.arange(len(normalised_data)), pre_post_peaks, normalised_data[pre_post_peaks])
    
    # Calculate the differences in x-coordinates (indices)
    delta_x = np.diff(pulse_peak_keys)

    # Calculate the differences in y-coordinates (amplitudes)
    delta_y = np.diff(normalised_data[pulse_peak_keys])

    # Calculate the slope angles in radians
    slope_angles = np.arctan(delta_y / delta_x)

    # Convert the slope angles from radians to degrees if needed
    slope_angles = np.degrees(slope_angles)

    # Get the mean of the slope angles
    mean_slope_angle = np.mean(slope_angles)

    # Get the standard deviation of the slope angles
    std_slope_angle = np.std(slope_angles)

    # Check if the standard deviation is close to zero
    if np.isclose(std_slope_angle, 0):
        # Handle the case when the standard deviation is zero or negligible
        upper_bound = mean_slope_angle
        lower_bound = mean_slope_angle
    else:
        # Define the upper and lower bounds for the slope angles
        upper_bound = mean_slope_angle + std_slope_angle
        lower_bound = mean_slope_angle - std_slope_angle

    print(upper_bound)
    print(lower_bound)


    anomal_peaks = []
    # Iterate over the slope_angles
    for i, angle in enumerate(slope_angles):
        # If the angle is greater than the upper bound or less than the lower bound
        if angle > upper_bound or angle < lower_bound:
            # Remove the corresponding peak from the pulse_peak_keys
            # Find the associated pulse peak
            pulse_peak = pulse_peak_keys[i]
            # Remove the pulse peak from the pulse_peak_keys
            anomal_peaks.append(pulse_peak)
            pulse_peak_keys.pop(i)

    if debug == True:
        # Add title
        # Create a subplot of two plots stacked on top of each other
        plt.subplot(2,1,1)
        plt.title("PULSE ONSET AND END")
        plt.plot(normalised_data)
        plt.plot(data_savgol)
        # Plot the current peak
        plt.plot(peaks, data_savgol[peaks], "gx")
        # Plot the candidate peak as a green dot on the norm data
        plt.plot(pulse_peak_keys, normalised_data[pulse_peak_keys], "go")
        # Plot the anomaly peaks as orange dots on the norm data
        plt.plot(anomal_peaks, normalised_data[anomal_peaks], "yo")
        # Plotting crossings_data as red dots
        plt.plot(pre_peaks, normalised_data[pre_peaks], "ro")
        plt.plot(post_peaks, normalised_data[post_peaks], "ro")
        # Plot crossings_norm_data as yellow dots
        #plt.plot(crossings_norm_data, d1_norm_data_filtered[crossings_norm_data], "yo")
        # Plot the pulse_centroids as blue dots
        plt.plot(pulse_centroid_keys, d1_norm_data_filtered[pulse_centroid_keys], "bo")
        # Plot the d2_norm_data_filtered_median as a black horizontal line
        plt.axhline(y=np.mean(normalised_data), color="k")
        # Plot the d1_norm_data_filtered
        plt.plot(d1_norm_data_filtered)

        # Plot the second subplot
        plt.subplot(2,1,2)
        # Plot the data's slope_angles as a histogram
        plt.hist(slope_angles, bins=100)
        # Create a title and add the upper and lower bounds to the title
        plt.title("SLOPE ANGLES\nUpper bound: {}\nLower bound: {}".format(upper_bound, lower_bound))

        plt.show()

        plt.plot(normalised_data)
        plt.plot(pulse_peak_keys, normalised_data[pulse_peak_keys], "go")
        plt.plot(pre_peaks, normalised_data[pre_peaks], "ro")
        plt.plot(post_peaks, normalised_data[post_peaks], "ro")
        plt.title("Detected peaks and troughs")
        plt.show()