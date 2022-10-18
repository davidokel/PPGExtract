from turtle import color
import pandas as pd
import numpy as np
import scipy.signal as sp
import math
import matplotlib.pyplot as plt
import random as rd
from upslopes_downslopes_rise_times_auc import *
from amplitudes_widths_prominences import *
import scipy.stats as stats

def run_test(data_df, window_seconds, fs, number_of_examples):
    columns = data_df.columns

    window_size = (window_seconds*fs)

    for i in range(number_of_examples): 
        random_patient = rd.randint(0,len(columns)-1)

        data = data_df[data_df.columns[random_patient]].dropna().to_numpy()
        data_flipped = np.flip(data)

        if len(data_flipped) !=0:
            random_chunk_start = rd.randint(0,len(data_flipped)-1)
            random_chunk_end = random_chunk_start + window_size
            data_chunk = data_flipped[random_chunk_start:random_chunk_end]

            plt.title("Patient " + str(random_patient) + " Chunk " + str(random_chunk_start) + ":" + str(random_chunk_start+window_size))
            plt.plot(data_chunk)
            manager = plt.get_current_fig_manager()
            manager.window.showMaximized()
            plt.show()
            
            chunk_filtered = normalise_data(data_chunk, fs)

            amplitudes, half_widths = get_amplitudes_widths_prominences(chunk_filtered,fs=100,visualise=1)
            upslope, downslope, rise_time, decay_time, auc, sys_auc, dia_auc, auc_ratio, second_derivative_ratio = get_upslopes_downslopes_rise_times_auc(chunk_filtered,fs=100,visualise=1)

# Removing values above a certain threshold from dataframe based on the gold_standard
def remove_values(gold_standard, distal, proximal, subtracted, upper=60, lower=0):
    
    # Iterate over columns of the gold_standard dataframe
    for column in range(len(gold_standard.columns)):
        gold_standard_data = np.array(gold_standard.iloc[:,column])
        distal_data = np.array(distal.iloc[:,column])
        proximal_data = np.array(proximal.iloc[:,column])
        subtracted_data = np.array(subtracted.iloc[:,column])

        upper_i = np.where(gold_standard_data > upper)[0]
        gold_standard_data = np.delete(gold_standard_data,upper_i)
        gold_standard.iloc[:,column] = pd.Series(gold_standard_data)

        distal_data = np.delete(distal_data,upper_i)
        distal.iloc[:,column] = pd.Series(distal_data)

        proximal_data = np.delete(proximal_data,upper_i)
        proximal.iloc[:,column] = pd.Series(proximal_data)

        subtracted_data = np.delete(subtracted_data,upper_i)
        subtracted.iloc[:,column] = pd.Series(subtracted_data)

        lower_i = np.where(gold_standard_data < lower)[0]
        gold_standard_data = np.delete(gold_standard_data,lower_i)
        gold_standard.iloc[:,column] = pd.Series(gold_standard_data)

        distal_data = np.delete(distal_data,lower_i)
        distal.iloc[:,column] = pd.Series(distal_data)

        proximal_data = np.delete(proximal_data,lower_i)
        proximal.iloc[:,column] = pd.Series(proximal_data)

        subtracted_data = np.delete(subtracted_data,lower_i)
        subtracted.iloc[:,column] = pd.Series(subtracted_data)

    print("Defined thresholds applied and values removed")
    return gold_standard, distal, proximal, subtracted
        
#########################
# LOADING DATA FROM CSV #
#########################
def load_csv(path):
    data = pd.read_csv(path)
    data.drop(data.columns[[0]],axis=1,inplace=True)
    return data

def normalise_data(data,fs):
    sos_ac = sp.butter(2, [0.5, 12], btype='bandpass', analog=False, output='sos', fs=fs) # Defining a high pass filter
    ac = sp.sosfiltfilt(sos_ac, data, axis=- 1, padtype='odd', padlen=None) # Applying the high pass filter to the data chunk
    
    sos_dc = sp.butter(3, (0.2/(fs/2)), btype='lowpass', analog=False, output='sos', fs=fs) # Defining a high pass filter
    dc = sp.sosfiltfilt(sos_dc, data, axis=- 1, padtype='odd', padlen=None) # Applying the high pass filter to the data chunk
    
    normalised = ac/dc

    return normalised

def band_pass_filter(data, order, fs, low_cut, high_cut):
    sos = sp.butter(order, [low_cut, high_cut], btype='bandpass', analog=False, output='sos', fs=fs) # Defining a high pass filter
    filtered_data = sp.sosfiltfilt(sos, data, axis=- 1, padtype='odd', padlen=None) # Applying the high pass filter to the data chunk
    return filtered_data

def cosine_rule(a, b, c):
    angle_c = math.acos((c**2+a**2-b**2)/(2*c*a))

    return angle_c

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

    """if len(anomalous_values) != 0:
        plt.plot(data)
        plt.plot(peak_locs, data[peak_locs], "x")
        plt.axhline(y=upper, color='r', linestyle='-')
        plt.axhline(y=lower, color='r', linestyle='-')
        # Iterating and plotting anomalous_values as dots
        for value in removed:
            plt.plot(value, data[value], "o")
        manager = plt.get_current_fig_manager()
        manager.window.showMaximized()
        plt.show()"""
    
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

        return peak_points