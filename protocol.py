import numpy as np
import matplotlib.pyplot as plt
from amplitudes_widths_prominences import *
from pulse_processing import *
from upslopes_downslopes_rise_times_auc import *
from data_methods import *
import math

clean_IICP = load_csv("C:/Users/k20113376/Documents/Clinical Trial/NEW_Code_Data/Data/Data_Cleaned/IMPROVED_iicp_data_cleaned_9.csv")
clean_810_distal = load_csv("C:/Users/k20113376/Documents/Clinical Trial/NEW_Code_Data/Data/Data_Cleaned/IMPROVED_DISTAL_810_nicp_data_cleaned_9.csv")
clean_810_proximal = load_csv("C:/Users/k20113376/Documents/Clinical Trial/NEW_Code_Data/Data/Data_Cleaned/IMPROVED_PROXIMAL_810_nicp_data_cleaned_9.csv")
clean_810_subtracted = load_csv("C:/Users/k20113376/Documents/Clinical Trial/NEW_Code_Data/Data/Data_Cleaned/IMPROVED_SUBTRACTED_810_nicp_data_cleaned_9.csv")

def run_protocol(window, iicp, distal, proximal, subtracted):
    num_windows = math.floor(len(iicp)/window)
    columns = clean_810_subtracted.columns

    for column in range(len(columns)):
        iicp_data = iicp[iicp.columns[column]].dropna()
        distal_data = distal[distal.columns[column]].dropna()
        proximal_data = proximal[proximal.columns[column]].dropna()
        subtracted_data = subtracted[subtracted.columns[column]].dropna()

        chunk_start = 0
        chunk_end = window
        for window in range(num_windows):
            iicp_chunk = iicp_data[chunk_start:chunk_end]
            distal_chunk = distal_data[chunk_start:chunk_end]
            proximal_chunk = proximal_data[chunk_start:chunk_end]
            subtracted_chunk = subtracted_data[chunk_start:chunk_end]

            iicp_peaks = pulse_detect(iicp_chunk,100,5,'d2max',vis = False)
            distal_peaks = pulse_detect(distal_chunk,100,5,'d2max',vis = False)
            proximal_peaks = pulse_detect(proximal_chunk,100,5,'d2max',vis = False)
            subtracted_peaks = pulse_detect(subtracted_chunk,100,5,'d2max',vis = False)

            iicp_amplitudes, iicp_half_widths = get_amplitudes_widths_prominences(iicp_chunk,fs=100,visualise=0)
            distal_amplitudes, distal_half_widths = get_amplitudes_widths_prominences(distal_chunk,fs=100,visualise=0)
            proximal_amplitudes, proximal_half_widths = get_amplitudes_widths_prominences(proximal_chunk,fs=100,visualise=0)
            subtracted_amplitudes, subtracted_half_widths = get_amplitudes_widths_prominences(subtracted_chunk,fs=100,visualise=0)

            iicp_upslopes, iicp_downslopes, iicp_rise_times, iicp_auc = get_upslopes_downslopes_rise_times_auc(iicp_chunk,fs=100,visualise=0)
            distal_upslopes, distal_downslopes, distal_rise_times, distal_auc = get_upslopes_downslopes_rise_times_auc(distal_chunk,fs=100,visualise=0)
            proximal_upslopes, proximal_downslopes, proximal_rise_times, proximal_auc = get_upslopes_downslopes_rise_times_auc(proximal_chunk,fs=100,visualise=0)
            subtracted_upslopes, subtracted_downslopes, subtracted_rise_times, subtracted_auc = get_upslopes_downslopes_rise_times_auc(subtracted_chunk,fs=100,visualise=0)

            chunk_start += window
            chunk_end += window

