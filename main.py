from cmath import nan
from data_methods import load_csv, band_pass_filter
import matplotlib.pyplot as plt
import random as rd
from amplitudes_widths_prominences import get_amplitudes_widths_prominences
from upslopes_downslopes_rise_times_auc import get_upslopes_downslopes_rise_times_auc
import numpy as np
from pulse_processing import *
from data_methods import *
from protocol import *

clean_IICP = load_csv("C:/Users/k20113376/Documents/Clinical Trial/NEW_Code_Data/Data/Data_Cleaned/IMPROVED_iicp_data_cleaned_9.csv")
clean_810_distal = load_csv("C:/Users/k20113376/Documents/Clinical Trial/NEW_Code_Data/Data/Data_Cleaned/IMPROVED_DISTAL_810_nicp_data_cleaned_9.csv")
clean_810_proximal = load_csv("C:/Users/k20113376/Documents/Clinical Trial/NEW_Code_Data/Data/Data_Cleaned/IMPROVED_PROXIMAL_810_nicp_data_cleaned_9.csv")
clean_810_subtracted = load_csv("C:/Users/k20113376/Documents/Clinical Trial/NEW_Code_Data/Data/Data_Cleaned/IMPROVED_SUBTRACTED_810_nicp_data_cleaned_9.csv")

clean_IICP, clean_810_distal, clean_810_proximal, clean_810_subtracted = remove_values(clean_IICP, clean_810_distal, clean_810_proximal, clean_810_subtracted)

fs = 100
window_size_seconds = 60
window_size_instances = fs*window_size_seconds

# 2,30,31,11
clean_IICP = clean_IICP.drop(columns = ['2','30','31','11'], axis=1)
clean_810_distal = clean_810_distal.drop(columns = ['2','30','31','11'], axis=1)
clean_810_proximal = clean_810_proximal.drop(columns = ['2','30','31','11'], axis=1)
clean_810_subtracted = clean_810_subtracted.drop(columns = ['2','30','31','11'], axis=1)

run_protocol(window_size_instances, clean_IICP, clean_810_distal, clean_810_proximal, clean_810_subtracted)

"""columns = clean_810_subtracted.columns

window_size = (5*100)

for i in range(40): 
    random_patient = rd.randint(0,len(columns)-1)
    #random_patient = 8
    data = clean_810_subtracted[clean_810_subtracted.columns[random_patient]].dropna().to_numpy()
    data_flipped = np.flip(data)

    if len(data_flipped) !=0:
        random_chunk_start = rd.randint(0,len(data_flipped)-1)
        random_chunk_end = random_chunk_start + window_size
        data_chunk = data_flipped[random_chunk_start:random_chunk_end]

        plt.title("Patient " + str(random_patient) + " Chunk " + str(random_chunk_start) + ":" + str(random_chunk_start+window_size))
        plt.plot(data_chunk)
        plt.show()
        
        chunk_filtered = band_pass_filter(data_chunk, 2, 100, 0.5, 12)
        normalised_chunk_filtered = (chunk_filtered - chunk_filtered.min())/(chunk_filtered.max() - chunk_filtered.min())

        amplitudes, half_widths = get_amplitudes_widths_prominences(normalised_chunk_filtered,fs=100,visualise=1)
        upslope, downslope, rise_time, decay_time, auc, sys_auc, dia_auc, auc_ratio, second_derivative_ratio = get_upslopes_downslopes_rise_times_auc(normalised_chunk_filtered,fs=100,visualise=1)"""