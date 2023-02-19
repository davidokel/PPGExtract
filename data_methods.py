import pandas as pd
import numpy as np
import scipy.signal as sp
import math
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


def normalise_data(data,fs):
    sos_ac = sp.butter(2, [0.5, 12], btype='bandpass', analog=False, output='sos', fs=fs) # Defining a high pass filter
    ac = sp.sosfiltfilt(sos_ac, data, axis= -1, padtype='odd', padlen=None) # Applying the high pass filter to the data chunk
    
    sos_dc = sp.butter(3, (0.2/(fs/2)), btype='lowpass', analog=False, output='sos', fs=fs) # Defining a high pass filter
    dc = sp.sosfiltfilt(sos_dc, data, axis= -1, padtype='odd', padlen=None) # Applying the high pass filter to the data chunk
    
    normalised = 10*(-(ac/dc))

    return normalised

def band_pass_filter(data, order, fs, low_cut, high_cut):
    sos = sp.butter(order, [low_cut, high_cut], btype='bandpass', analog=False, output='sos', fs=fs) # Defining a high pass filter
    filtered_data = sp.sosfiltfilt(sos, data, axis=- 1, padtype='odd', padlen=None) # Applying the high pass filter to the data chunk
    return filtered_data

def data_scaler(data):
    min_data = min(data)
    if min_data < 0:
        scaling_factor = abs(min_data)
    else:
        scaling_factor = np.diff([0, min_data])[0]

    data = data + scaling_factor
    return data

def find_anomalies(data):
    upper, lower, envelope_difference = get_envelope(data) # Get the upper and lower envelope of the data

    anomalies = {} # Defining a dictionary to store the detected anomalies of the signal
    anomly_segment_trigger = 1
    anomaly_count = 1

    for i in range(len(data)):
        # If anomly_segment_trigger = 1 then the code is "searching" for a new anomaly
        # If the anomly_segment_trigger = 0 then the code is "searching" for a local maximum or minimum anomaly
        # Check to if there have not been any anomalies detected yet
        if len(anomalies) == 0:
            # If the current index of the envelope difference is greater than or equal to the upper threshold
            # and less than or equal to the lower threshold then add the current index to the list of anomalies
            if data[i] >= upper or data[i] <= lower:
                anomalies[anomaly_count] = i
                anomly_segment_trigger = 0
        # Check to see if current value at index i is greater than or equal to the upper threshold or lower than or equal to the lower threshold
        elif data[i] >= upper or data[i] <= lower:
            # If the code is "searching" for a local maximum or minimum anomaly
            # check if the current value is greater than or less than the existing anomaly
            # if it is greater than the existing above the upper threshold anomaly then replace it with the current index
            # if it is less than the existing below the lower threshold anomaly then replace it with the current index
            if anomly_segment_trigger == 0:
                if data[anomalies[anomaly_count]] >= upper and data[i] > data[anomalies[anomaly_count]]:
                    anomalies[anomaly_count] = i
                if data[anomalies[anomaly_count]] <= lower and data[i] < data[anomalies[anomaly_count]]:
                    anomalies[anomaly_count] = i
            # If the code is "searching" for a new anomaly
            # add the anomaly to the anomalies dictionary
            elif anomly_segment_trigger == 1:
                anomaly_count += 1
                anomalies[anomaly_count] = i
                anomly_segment_trigger = 0
        # If the value at the current index is less than the upper threshold and greater than the lower threshold
        # then set the anomly_segment_trigger to 1
        elif data[i] < upper and data[i] > lower:
            anomly_segment_trigger = 1
        
    return anomalies

def get_peaks(data,fs):
    # Convertion to list
    peaks = []

    data = (data - data.min())/(data.max() - data.min())
    
    height = np.percentile(data, 70)
    peak_locs,_ = sp.find_peaks(data, height=height, distance=fs*0.4, prominence=0.1)

    q3, q1 = np.percentile(data, [75, 25])
    iqr = q3 - q1
    
    upper = q3 + (2*iqr)
    lower = q1 - (2*iqr)

    peak_locs = peak_locs.tolist()

    anomalous_values = find_anomalies(data, upper, lower)

    if len(peak_locs) != 0:
    
        removed = []
        for key, value in anomalous_values.items():
            if value in peak_locs:
                closest_peak_index = peak_locs.index(value)
            else:
                absolute_differences = lambda list_value : abs(list_value - value)
                closest_value = min(peak_locs, key=absolute_differences)
                closest_peak_index = peak_locs.index(closest_value)

            # IF PEAK IS NOT THE FRIST OR LAST PEAK
            if closest_peak_index != 0 and closest_peak_index != len(peak_locs)-1:
                removed.append(peak_locs[closest_peak_index-1])
                removed.append(peak_locs[closest_peak_index])
                removed.append(peak_locs[closest_peak_index+1])

            # IF THE PEAK IS THE FIRST PEAK BUT NOT THE LAST PEAK
            elif closest_peak_index == 0 and closest_peak_index != len(peak_locs)-1:
                removed.append(peak_locs[closest_peak_index])
                removed.append(peak_locs[closest_peak_index+1])
            
            # IF THE PEAK IS THE LAST PEAK BUT NOT THE FIRST PEAK
            elif closest_peak_index == len(peak_locs)-1 and closest_peak_index != len(peak_locs)-1:
                removed.append(peak_locs[closest_peak_index-1])
                removed.append(peak_locs[closest_peak_index])
            
            # IF THE PEAK IS THE FIRST AND LAST PEAK
            elif closest_peak_index == 0 and closest_peak_index == len(peak_locs)-1:
                removed.append(peak_locs[closest_peak_index])

        peak_locs = [i for i in peak_locs if i not in removed]

        if len(peak_locs) > 2:
            del peak_locs[0]
            del peak_locs[-1]

        prominences_all = (sp.peak_prominences(data, peak_locs)[0]).tolist()
        widths_all = sp.peak_widths(data, peak_locs)[0].tolist()
        
        for peak in range(len(peak_locs)):
            # Calculating the search space either side of the peak:
            # The search space is 50% of the prominence of the peak
            # The prominence is multiplied by 100 for scaling purposes
            search_distance = abs(int((widths_all[peak])))

            # Applying the search space to the peak location
            data_start = int((peak_locs[peak]) - search_distance)
            data_end = int((peak_locs[peak]) + search_distance)

            # Handling if the search space is out of bounds
            if data_start < 0:
                data_start = 0
            if data_end > len(data):
                data_end = len(data)-1

            if peak != 0 and peak != len(peak_locs)-1:
                # Handling if start point is less than pervious peak
                if data_start <= peak_locs[peak-1]:
                    current = peak_locs[peak]
                    previous = peak_locs[peak-1]
                    difference = current - previous
                    dfference_2 = int(difference/2)
                    data_start = math.floor(current - dfference_2)
                # Handling if end point is greater than next peak
                if data_end >= peak_locs[peak+1]:
                    current = peak_locs[peak]
                    next = peak_locs[peak+1]
                    difference = next - current
                    dfference_2 = int(difference/2)
                    data_end = math.floor(current + dfference_2) 
            if peak == 0 and len(peak_locs) > 1:
                # Handling if end point is greater than next peak
                if data_end >= peak_locs[peak+1]:
                    # Calculating difference (number of instances between current peak and next peak)
                    current = peak_locs[peak]
                    next = peak_locs[peak+1]
                    difference = next - current
                    dfference_2 = int(difference/2)
                    # Set data end as the current peak + half the difference
                    data_end = math.floor(peak_locs[peak] + dfference_2)
            
            if len(data[data_start:peak_locs[peak]]) > 0:
                pre_data = np.mean(data[data_start:peak_locs[peak]])
            else:
                print("Error: No data in pre peak range")
            
            if len(data[peak_locs[peak]:data_end]) > 0:
                post_data = np.mean(data[peak_locs[peak]:data_end])
            else:
                print("Error: No data in post peak range")

            if (abs((data[peak_locs[peak]]-pre_data)/pre_data)*100) > 5 and (abs((data[peak_locs[peak]]-post_data)/post_data)*100) > 5:
                peaks.append(math.floor(peak_locs[peak]))
    
    return peaks

def get_onsets_v2(data,fs=100,debug=False):
    # Normalising the data
    data_raw = data
    #data = normalise_data(data,fs)

    # Filter data using savgol filter
    data = sp.savgol_filter(data, 51, 3)

    # Calculate the first and second derivative of the data
    data_first_derivative = np.diff(data)
    data_second_derivative = np.diff(data_first_derivative)

    # Add a zero to the first and second derivative as the first and second derivative are one index shorter than the original data
    data_first_derivative = np.insert(data_first_derivative, 0, 0)
    data_second_derivative = np.insert(data_second_derivative, 0, 0)
    data_second_derivative = np.insert(data_second_derivative, 0, 0)

    # Compute the element-wise difference between the two arrays
    diff = data_first_derivative - data_second_derivative
    
    # Identify the indices where the sign of the difference changes
    sign_change = np.where(np.diff(np.sign(diff)))[0]
    
    # Add one to the indices to account for the offset introduced by np.diff
    crossings = sign_change + 1

    # With an fs of 100Hz calculate what 220bpm equates to in terms of samples
    bpm_220 = int((60/220)*fs)

    # Remove any crossings that are less than 220bpm apart
    crossings = [i for i in crossings if i > bpm_220]

    # Finding the peaks from the crossings
    peaks = []
    troughs = []
    notches = []
    peak_points = {}

    # If there are no crossings then return an empty list
    if len(crossings) == 0:
        return peaks
    else:
        # Iterate over the crossing points starting from the second crossing and ending at the second to last crossing
        for i in range(1, len(crossings)-1):
            pre_data_slope = get_signal_slopes(data, crossings[i-1], crossings[i])
            post_data_slope = get_signal_slopes(data, crossings[i], crossings[i+1])

            # If the slope of the data before the crossing is positive and the slope of the data after the crossing is negative then the crossing is a peak
            if pre_data_slope > 0 and post_data_slope < 0:
                peaks.append(crossings[i])

            # If the slope of the data before the crossing is negative and the slope of the data after the crossing is positive then the crossing is a trough
            if pre_data_slope < 0 and post_data_slope > 0:
                troughs.append(crossings[i])

    if debug:
        plt.plot(data)
        plt.title("CLASSIFY PEAKS AND TROUGHS")
        plt.plot(peaks, data[peaks], "gx")
        plt.plot(troughs, data[troughs], "rx")
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
            
        if peaks[i] + search_space > len(data_raw):
            peak_explore_end = len(data_raw)
        else:
            peak_explore_end = peaks[i] + search_space

        # Find the local maxima and minima
        peak_raw = np.argmax(data_raw[peak_explore_start:peak_explore_end]) + peak_explore_start

        # Add the local maxima and minima to the list
        peaks_raw.append(peak_raw)

    for i in range(0, len(troughs)):
        # Define the exploration space for the trough
        if troughs[i] - search_space < 0:
            trough_explore_start = 0
        else:
            trough_explore_start = troughs[i] - search_space

        if troughs[i] + search_space > len(data_raw):
            trough_explore_end = len(data_raw)
        else:
            trough_explore_end = troughs[i] + search_space

        trough_raw = np.argmin(data_raw[trough_explore_start:trough_explore_end]) + trough_explore_start
        
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
            # Get the values of the data_raw at the peaks between the current trough and the next trough
            peaks_between_troughs_values = [data_raw[peak] for peak in peaks_between_troughs]

            # Get the index of the local maxima
            local_maxima_index = np.argmax(peaks_between_troughs_values)

            # Remove the peaks that are not the local maxima
            for j, peak in enumerate(peaks_between_troughs):
                if j != local_maxima_index:
                    peaks_raw.remove(peak)

    if debug:
        plt.plot(data_raw)
        plt.title("PRE-DICTIONARY")
        plt.plot(peaks_raw, data_raw[peaks_raw], "gx")
        plt.plot(troughs_raw, data_raw[troughs_raw], "rx")
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

                    peak_points[i] = {"Peak": peak, "Pre_peak": pre_peak, "Post_peak": post_peak}
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

                    if post_peak < peaks_raw[i+1]:
                        peak_points[i] = {"Peak": peak, "Pre_peak": pre_peak, "Post_peak": post_peak}
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
                    
                    # Ensure that the pre_peak is >= the previous peak's post_peak and the post_peak is not greater than the next peak
                    if pre_peak >= peak_points[last_key]["Post_peak"] and post_peak < peaks_raw[i+1]:
                        # Add the peaks and troughs to the dictionary
                        peak_points[i] = {"Peak": peak, "Pre_peak": pre_peak, "Post_peak": post_peak}
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
                    
                    # Ensure that the pre_peak is >= the previous peak's post_peak
                    if pre_peak >= peak_points[last_key]["Post_peak"]:
                        # Add the peaks and troughs to the dictionary
                        peak_points[i] = {"Peak": peak, "Pre_peak": pre_peak, "Post_peak": post_peak}
                else:
                    continue

        # Get all the pre and post peak points from the dictionary and store them in a list
        troughs = []
        peaks = []
        for key in peak_points:
            troughs.append(peak_points[key]["Pre_peak"])
            troughs.append(peak_points[key]["Post_peak"])
            
            # Get the index of the maximum value between the pre_peak and post_peak points
            corrected_peak = np.argmax(data_raw[peak_points[key]["Pre_peak"]:peak_points[key]["Post_peak"]]) + peak_points[key]["Pre_peak"]
            
            # Change the peak value in the dictionary to the maximum value between the pre_peak and post_peak
            peak_points[key]["Peak"] = corrected_peak

            peaks.append(peak_points[key]["Peak"])

        # Plot the data and the peaks and troughs
        if debug:
            plt.plot(data_raw)
            plt.plot(peaks, data_raw[peaks], "go")
            plt.plot(troughs, data_raw[troughs], "ro")
            plt.show()

    return peak_points, peaks, troughs

def get_signal_slopes(data,index1,index2):
        # Compute the difference in y-values and x-values
    y_diff = data[index2] - data[index1]
    x_diff = index2 - index1
    
    # Compute the slope
    slope = y_diff / x_diff
    
    return slope

def get_onsets(data,peak_locs,fs=100):
    data = (data - data.min())/(data.max() - data.min())
    data_list = data.tolist()
    peak_points = {}
    pre_peaks = []
    post_peaks = []
    
    if len(peak_locs) > 0:
        # If peaks exist calculate all prominences for the exisitng peaks
        prominences_all = (sp.peak_prominences(data_list, peak_locs)[0]).tolist()
        widths_all = sp.peak_widths(data, peak_locs)[0].tolist()

        # Iterating through all the peaks
        for peak in range(len(peak_locs)-1):
                # Calculating the search space either side of the peak:
                # The search space is 80% of the prominence of the peak
                # The prominence is multiplied by 100 for scaling purposes
                #search_distance = abs(int((prominences_all[peak]*100)*2))
                search_distance = abs(int((widths_all[peak]))*2)

                # Applying the search space to the peak location
                data_start = int((peak_locs[peak]) - search_distance)
                data_end = int((peak_locs[peak]) + search_distance)
                
                # Handling if the search space is out of bounds
                if data_start <0:
                    data_start = 0
                if data_end > len(data_list):
                    data_end = len(data_list)-1
                
                # If not the first or last peak
                if peak != 0 and peak != len(peak_locs)-1:
                    keys = list(peak_points.keys())
                    # Handling if start point is less than previous post peak point
                    if data_start < peak_points["Peak_"+str(peak-1)]["Post_Peak"]:
                        #"Peak_"+str(peak)
                        # Set data start as the previous post peak point
                        data_start = peak_points["Peak_"+str(peak-1)]["Post_Peak"]
                    # Handling if end point is greater than next peak
                    if data_end >= peak_locs[peak+1]:
                        # Calculating difference (number of instances between current peak and next peak)
                        current = peak_locs[peak]
                        next = peak_locs[peak+1]
                        difference = next - current
                        dfference_2 = int(difference/2)
                        # Set data end as the current peak + half the difference
                        #data_end = math.floor(peak_locs[peak] + dfference_2)
                        data_end = peak_locs[peak+1]
                # If first peak and more than one peak
                if peak == 0 and len(peak_locs) > 1:
                    # Handling if end point is greater than next peak
                    if data_end >= peak_locs[peak+1]:
                        # Calculating difference (number of instances between current peak and next peak)
                        current = peak_locs[peak]
                        next = peak_locs[peak+1]
                        difference = next - current
                        dfference_2 = int(difference/2)
                        # Set data end as the current peak + half the difference
                        data_end = math.floor(peak_locs[peak] + dfference_2)

                if len(data_list[data_start:peak_locs[peak]]) > 0:
                    min_pre = math.floor(data_list.index(min(data_list[data_start:peak_locs[peak]])))
                else:
                    print("Error: No data in pre peak range")
                
                if len(data_list[peak_locs[peak]:data_end]) > 0:
                    min_post = math.floor(data_list.index(min(data_list[peak_locs[peak]:data_end])))
                else:
                    print("Error: No data in post peak range")
                
                peak_points["Peak_"+str(peak)] = {}
                peak_points["Peak_"+str(peak)]["Peak"] = peak_locs[peak]
                peak_points["Peak_"+str(peak)]["Pre_Peak"] = min_pre
                peak_points["Peak_"+str(peak)]["Post_Peak"] = min_post
                pre_peaks.append(min_pre)
                post_peaks.append(min_post)
   
        return peak_points

def get_envelope(data,seconds,fs):
    distance = (len(data)/fs)*seconds
    
    if distance < 1:
        distance = 1

    data = normalise_data(data,fs)

    _,peaks,troughs = get_onsets_v2(data, 100) 
    #troughs = get_onsets_v2(-data, 100)

    if len(peaks) > 4 and len(troughs) > 4:
        u_p = interp1d(peaks, data[peaks], kind = 'cubic', bounds_error=False, fill_value = np.median(data[peaks]))
        l_p = interp1d(troughs, data[troughs],  kind = 'cubic', bounds_error=False, fill_value = np.median(data[troughs]))
       
        upper_envelope = np.array([u_p(i) for i in range(len(data))])
        lower_envelope = np.array([l_p(i) for i in range(len(data))])
    else:
        upper_envelope = []
        lower_envelope = []
        peaks = []
        troughs = []
    
    if len(upper_envelope) > 0 or len(lower_envelope) > 0:
        envelope_difference = upper_envelope - lower_envelope
    else:
        envelope_difference = []

    # Plot the data and the envelopes
    plt.plot(data)
    plt.plot(upper_envelope)
    plt.plot(lower_envelope)
    # Plot the peaks and troughs
    plt.plot(peaks, data[peaks], "x")
    plt.plot(troughs, data[troughs], "x")
    plt.show()


    return upper_envelope, lower_envelope, envelope_difference