import pandas as pd
import numpy as np
import scipy.signal as sp
import math
from scipy.interpolate import interp1d

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

def find_anomalies(data, upper, lower):
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

        #peaks = peak_locs

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

    peaks = get_peaks(data, 100) 
    troughs = get_peaks(-data, 100)

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

    return upper_envelope, lower_envelope, peaks, troughs, envelope_difference