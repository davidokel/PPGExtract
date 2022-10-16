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
import scipy.signal as sp

#clean_IICP = load_csv("Data_Test/IMPROVED_V2_line_threshold_0.0025_threshold_2_iicp_data_cleaned_9.csv")
#clean_810_distal = load_csv("Data_Test/IMPROVED_V2_line_threshold_0.0025_threshold_2_DISTAL_810_nicp_data_cleaned_9.csv")
#clean_810_proximal = load_csv("Data_Test/IMPROVED_V2_line_threshold_0.0025_threshold_2_PROXIMAL_810_nicp_data_cleaned_9.csv")
#clean_810_subtracted = load_csv("Data_Test/IMPROVED_V2_line_threshold_0.0025_threshold_2_SUBTRACTED_810_nicp_data_cleaned_9.csv")

clean_IICP = load_csv("C:/Users/k20113376/Documents/Clinical Trial/NEW_Code_Data/Data/Data_Cleaned/IMPROVED_iicp_data_cleaned_9.csv")
clean_810_distal = load_csv("C:/Users/k20113376/Documents/Clinical Trial/NEW_Code_Data/Data/Data_Cleaned/IMPROVED_DISTAL_810_nicp_data_cleaned_9.csv")
clean_810_proximal = load_csv("C:/Users/k20113376/Documents/Clinical Trial/NEW_Code_Data/Data/Data_Cleaned/IMPROVED_PROXIMAL_810_nicp_data_cleaned_9.csv")
clean_810_subtracted = get_subtracted_data(clean_810_distal,clean_810_proximal)
#clean_810_subtracted = load_csv("C:/Users/k20113376/Documents/Clinical Trial/NEW_Code_Data/Data/Data_Cleaned/IMPROVED_SUBTRACTED_810_nicp_data_cleaned_9.csv")

print(clean_810_distal.head())
print(clean_810_proximal.head())
print(clean_810_subtracted.head())

print(len(clean_810_distal))
print(len(clean_810_proximal))
print(len(clean_810_subtracted))

run_test(clean_810_subtracted, 60, 100, 10)

clean_IICP, clean_810_distal, clean_810_proximal, clean_810_subtracted = remove_values(clean_IICP, clean_810_distal, clean_810_proximal, clean_810_subtracted)

fs = 100
window_size_seconds = 60
window_size_instances = fs*window_size_seconds

# 2,30,31,11
clean_IICP = clean_IICP.drop(columns = ['2','30','31','11'], axis=1)
clean_810_distal = clean_810_distal.drop(columns = ['2','30','31','11'], axis=1)
clean_810_proximal = clean_810_proximal.drop(columns = ['2','30','31','11'], axis=1)
clean_810_subtracted = clean_810_subtracted.drop(columns = ['2','30','31','11'], axis=1)

#run_protocol(window_size_instances, clean_IICP, clean_810_distal, clean_810_proximal, clean_810_subtracted)