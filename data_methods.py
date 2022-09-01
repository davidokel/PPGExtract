from turtle import color
import pandas as pd
import numpy as np
import scipy.signal as sp
import math
import matplotlib.pyplot as plt

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

    peak_locs,_ = sp.find_peaks(data,distance=(fs*0.3), prominence=prominence)
    half_widths_all = sp.peak_widths(data, peak_locs, rel_height=0.5)[0].tolist()
    prominences_all = (sp.peak_prominences(data, peak_locs)[0]).tolist()
    peak_locs.tolist()
    peaks = []

    for peak in range(len(peak_locs)):
        data_start = int(peak_locs[peak] - ((prominences_all[peak]*100)/2))
        data_end = int(peak_locs[peak] + ((prominences_all[peak]*100)/2))

        if peak != 0 and peak != len(peak_locs)-1:
            if data_start < peak_locs[peak-1]:
                data_start = peak_locs[peak-1]
            if data_end > peak_locs[peak+1]:
                data_end = peak_locs[peak+1]
        if peak == len(peak_locs)-1:
            print("entered")
            if data_end > len(data):
                data_end = len(data)
        if peak == 0:
            if data_start < 0:
                data_start = 0
        
        pre_data = np.mean(data_list[data_start:peak_locs[peak]])
        post_data = np.mean(data_list[peak_locs[peak]:data_end])

        if ((data[peak_locs[peak]]-pre_data)/pre_data)*100 > 20 and ((data[peak_locs[peak]]-post_data)/post_data)*100 > 20 and int(half_widths_all[peak]) < int(prominences_all[peak]*100):
            peaks.append(peak_locs[peak])
        """for peak in range(len(peak_locs)):
            if int(half_widths_all[peak]) < int(prominences_all[peak]*100):
                peaks.append(peak_locs[peak])"""

    return peaks

def get_onsets(data,peak_locs):
    data_list = data.tolist()
    peak_points = {}

    half_widths_all = sp.peak_widths(data_list, peak_locs, rel_height=0.5)[0].tolist()
    prominences_all = (sp.peak_prominences(data_list, peak_locs)[0]).tolist()
    print(prominences_all)

    if len(peak_locs) > 0:
        for peak in range(len(peak_locs)):
                data_start = int(peak_locs[peak] - ((prominences_all[peak]*100)*0.8))
                data_end = int(peak_locs[peak] + ((prominences_all[peak]*100)*0.8))

                if peak != 0 and peak != len(peak_locs)-1:
                    if data_start < peak_locs[peak-1]:
                        data_start = peak_locs[peak-1]
                    if data_end > peak_locs[peak+1]:
                        data_end = peak_locs[peak+1]
                if peak == len(peak_locs)-1:
                    print("entered")
                    if data_end > len(data):
                        data_end = len(data)
                if peak == 0:
                    if data_start < 0:
                        data_start = 0
                
                min_pre = data_list.index(min(data_list[data_start:peak_locs[peak]]))
                min_post = data_list.index(min(data_list[peak_locs[peak]:data_end]))
                
                peak_points["Peak_"+str(peak)] = {}
                peak_points["Peak_"+str(peak)]["Peak"] = peak_locs[peak]
                peak_points["Peak_"+str(peak)]["Pre_Peak"] = min_pre
                peak_points["Peak_"+str(peak)]["Post_Peak"] = min_post

        return peak_points