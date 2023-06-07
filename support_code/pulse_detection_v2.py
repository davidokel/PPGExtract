import matplotlib.pyplot as plt
from support_code.data_methods import normalise_data, get_signal_slopes
import numpy as np
import scipy.signal as sp
from operator import itemgetter
from scipy.signal import peak_widths
# Import butterworth filter
from scipy.signal import butter

def get_pulses(data,fs=100,visualise=False,debug=False):
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

    # Plot the raw data and the moving average
    if visualise:
        plt.plot(data, label='Raw Data')
        plt.plot(moving_average_data, label='Moving Average')
        plt.legend()
        plt.show()

    normalised_data = normalise_data(data,fs)

    # Filter data using savgol filter
    data_savgol = sp.savgol_filter(normalised_data, 103, 5)
        
    # Calculate the first and second derivative of the data
    # Using np.gradient() to calculate the first derivative and second derivative
    # Using np.gradient instead of np.diff as np.gradient returns the same number of elements as the original array
    d1_norm_data = np.gradient(normalised_data)
    d2_norm_data = np.gradient(d1_norm_data)

    # Filter d1 and d2 using savgol filter
    d1_norm_data_filtered = sp.savgol_filter(d1_norm_data, 47, 5)
    d2_norm_data_filtered = sp.savgol_filter(d2_norm_data, 47, 5)
    
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

    # Finding the peaks from the crossings
    peaks = []
    troughs = []
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
        # Define two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        
        # In the first one plot the savgold filtered data and the crossings
        ax1.plot(data_savgol)
        ax1.plot(crossings_savgol_data, data_savgol[crossings_savgol_data], "gx")
        ax1.set_title("Filtered Data and Crossings")
        
        # In the second subplot plot the first and second derivative of the savgol filtered data multiplied by 5, the savgold filtered data and the peaks and troughs
        ax2.plot(data_savgol)
        ax2.plot(peaks, data_savgol[peaks], "gx")
        ax2.plot(troughs, data_savgol[troughs], "rx")
        ax2.set_title("Filtered Data (Classified Peaks and Troughs)")
        
        plt.show()

    # Generate a dictionary called peak_points for storing peak information
    peak_points = {}

    # Iterate over the peaks
    for i in range(0, len(peaks)):

        """diff_data = normalised_data - np.mean(normalised_data)
        crossings_data = np.where(np.gradient(np.sign(diff_data)))[0]"""
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

                # Get the width of the pulse
                pulse_width = int(abs(left_crossing - right_crossing))

                pulse_onset_search_start = int(pulse_onset_search - pulse_width)
                pulse_onset_search_end = left_crossing

                pulse_end_search_start = right_crossing
                pulse_end_search_end = int(pulse_end_search + pulse_width)

                """# Get distance between the left_crossing and the centroid
                distance_left_crossing_centroid = int(abs(pulse_centroid - left_crossing))
                # Get distance between the right_crossing and the centroid
                distance_right_crossing_centroid = int(abs(pulse_centroid - right_crossing))

                pulse_onset_search_start = int(pulse_onset_search - distance_left_crossing_centroid)
                pulse_onset_search_end = left_crossing

                pulse_end_search_start = right_crossing
                pulse_end_search_end = int(pulse_end_search + distance_right_crossing_centroid)"""
            
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

                """if debug == True:
                    plt.plot(normalised_data)
                    plt.plot(pulse_onset, normalised_data[pulse_onset], "ro")
                    plt.plot(pulse_end, normalised_data[pulse_end], "ro")
                    plt.plot(pulse_peak, normalised_data[pulse_peak], "go")
                    plt.plot(pulse_centroid, normalised_data[pulse_centroid], "bo")
                    # Plot the left and right crossings
                    plt.plot(left_crossing, normalised_data[left_crossing], "yo")
                    plt.plot(right_crossing, normalised_data[right_crossing], "yo")
                    # Plot the search spaces as vertical dashed lines
                    plt.axvline(x=pulse_onset_search_start, color="k", linestyle="--")
                    plt.axvline(x=pulse_onset_search_end, color="k", linestyle="--")
                    plt.axvline(x=pulse_end_search_start, color="k", linestyle="--")
                    plt.axvline(x=pulse_end_search_end, color="k", linestyle="--")
                    plt.plot(np.mean(normalised_data), "k--")
                    plt.show()"""
                    

                # Update the peak_points dictionary with the pulse_onset, pulse_end and pulse_peak
                peak_points[key]["Peak"] = pulse_peak
                peak_points[key]["Relative_peak"] = pulse_peak - pulse_onset
                peak_points[key]["Raw_pulse_data"] = data[pulse_onset:pulse_end]
                peak_points[key]["Norm_pulse_data"] = normalised_data[pulse_onset:pulse_end]
                peak_points[key]["Pre_peak"] = pulse_onset
                peak_points[key]["Post_peak"] = pulse_end
        else:
            continue

    # Use itemgetter to get the "Peak" value from the peak_points dictionary
    peak_points_values = list(map(itemgetter("Peak"), peak_points.values()))
    # Use itemgetter to get the "Pre_peak" value from the peak_points dictionary
    peak_points_pre_peak_values = list(map(itemgetter("Pre_peak"), peak_points.values()))
    # Use itemgetter to get the "Post_peak" value from the peak_points dictionary
    peak_points_post_peak_values = list(map(itemgetter("Post_peak"), peak_points.values()))
    # Use itemgetter to get the "Pulse_centroid" value from the peak_points dictionary
    peak_points_pulse_centroid_values = list(map(itemgetter("Pulse_centroid"), peak_points.values()))
    # Combine the peak_points_pre_peak_values and peak_points_post_peak_values
    peak_points_pre_post_values = peak_points_pre_peak_values + peak_points_post_peak_values

    # If debug is True
    if debug == True:
        plt.plot(normalised_data)
        plt.plot(data_savgol)
        # plot the np.mean of the normalised_data as a black dashed line
        plt.axhline(y=np.mean(normalised_data), color="k", linestyle="--")
        plt.plot(crossings_data, normalised_data[crossings_data], "ko")
        plt.plot(peaks, data_savgol[peaks], "gx")
        plt.plot(peak_points_values, normalised_data[peak_points_values], "go")
        plt.plot(peak_points_pre_post_values, normalised_data[peak_points_pre_post_values], "ro")
        plt.plot(crossings_norm_data, d1_norm_data_filtered[crossings_norm_data], "yo")
        plt.plot(peak_points_pulse_centroid_values, d1_norm_data_filtered[peak_points_pulse_centroid_values], "bo")
        plt.plot(d1_norm_data_filtered, "g-")
        plt.plot(d2_norm_data_filtered, "r-")
        plt.show()

    if visualise == True:
        # Create two subplots and stack them vertically
        fig, axs = plt.subplots(2, 1, sharex=True)

        # Plot the data
        axs[0].set_title("Raw Data")
        # Convert data from a list to a numpy array
        plot_data = np.array(data)
        # Multiply the data by -1 to invert the data
        plot_data = plot_data * -1
        # Add the content to each subplot (replace with your own data)
        axs[0].plot(plot_data)
        
        # Plot the data
        axs[1].set_title("Normalised Data and Detected Pulses")
        # Add the content to each subplot (replace with your own data)
        axs[1].plot(normalised_data)
        # Plot the peak_points_values on the data as green dots
        axs[1].plot(peak_points_values, normalised_data[peak_points_values], "go")
        # Plot the peak_points_pre_post_values on the data as red dots
        axs[1].plot(peak_points_pre_post_values, normalised_data[peak_points_pre_post_values], "ro")
        
        plt.show()