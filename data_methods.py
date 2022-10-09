from turtle import color
import pandas as pd
import numpy as np
import scipy.signal as sp
import math
import matplotlib.pyplot as plt

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

def band_pass_filter(data, order, fs, low_cut, high_cut):
    sos = sp.butter(order, [low_cut, high_cut], btype='bandpass', analog=False, output='sos', fs=fs) # Defining a high pass filter
    filtered_data = sp.sosfiltfilt(sos, data, axis=- 1, padtype='odd', padlen=None) # Applying the high pass filter to the data chunk
    return filtered_data

def cosine_rule(a, b, c):
    angle_c = math.acos((c**2+a**2-b**2)/(2*c*a))

    return angle_c

def get_peaks(data,fs,prominence=0.4):
    data_list = data.tolist()
    peak_locs,_ = sp.find_peaks(data,distance=(fs*0.3),prominence=prominence)
    half_widths_all = sp.peak_widths(data, peak_locs, rel_height=0.5)[0].tolist()
    prominences_all = (sp.peak_prominences(data, peak_locs)[0]).tolist()
    peak_locs.tolist()
    peaks = []

    for peak in range(len(peak_locs)):
        # Calculating the search space either side of the peak:
        # The search space is 50% of the prominence of the peak
        # The prominence is multiplied by 100 for scaling purposes
        search_distance = abs(int((prominences_all[peak]*100)/2))

        # Applying the search space to the peak location
        data_start = int((peak_locs[peak]) - search_distance)
        data_end = int((peak_locs[peak]) + search_distance)

        # Handling if the search space is out of bounds
        if data_start <0:
            data_start = 0
        if data_end > len(data_list):
            data_end = len(data_list)-1

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
        
        if len(data_list[data_start:peak_locs[peak]]) > 0:
            pre_data = np.mean(data_list[data_start:peak_locs[peak]])
        else:
            print("Error: No data in pre peak range")
        
        if len(data_list[peak_locs[peak]:data_end]) > 0:
            post_data = np.mean(data_list[peak_locs[peak]:data_end])
        else:
            print("Error: No data in post peak range")

        if ((data[peak_locs[peak]]-pre_data)/pre_data)*100 > 10 and ((data[peak_locs[peak]]-post_data)/post_data)*100 > 10 and int(half_widths_all[peak]) < int(prominences_all[peak]*100):
            peaks.append(math.floor(peak_locs[peak]))

    return peaks

def get_onsets(data,peak_locs):
    data_list = data.tolist()
    peak_points = {}
    
    if len(peak_locs) > 0:
        # If peaks exist calculate all prominences for the exisitng peaks
        prominences_all = (sp.peak_prominences(data_list, peak_locs)[0]).tolist()

        # Iterating through all the peaks
        for peak in range(len(peak_locs)):
                # Calculating the search space either side of the peak:
                # The search space is 80% of the prominence of the peak
                # The prominence is multiplied by 100 for scaling purposes
                search_distance = abs(int((prominences_all[peak]*100)*0.8))

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
                    if data_start < peak_points[keys[-1]]["Post_Peak"]:
                        # Set data start as the previous post peak point
                        data_start = peak_points[keys[-1]]["Post_Peak"]
                    # Handling if end point is greater than next peak
                    if data_end >= peak_locs[peak+1]:
                        # Calculating difference (number of instances between current peak and next peak)
                        current = peak_locs[peak]
                        next = peak_locs[peak+1]
                        difference = next - current
                        dfference_2 = int(difference/2)
                        # Set data end as the current peak + half the difference
                        data_end = math.floor(peak_locs[peak] + dfference_2)
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