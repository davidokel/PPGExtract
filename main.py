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
from protocol_updated import *

clean_IICP = load_csv("C:/Users/k20113376/Documents/Clinical Trial/NEW_Code_Data/Data/Data_Cleaned/Cleaned_data_V4/IMPROVED_V4_line_threshold_0.0025_threshold_1.5_iicp_data_cleaned_9.csv")
clean_810_distal = load_csv("C:/Users/k20113376/Documents/Clinical Trial/NEW_Code_Data/Data/Data_Cleaned/Cleaned_data_V4/IMPROVED_V4_line_threshold_0.0025_threshold_1.5_DISTAL_810_nicp_data_cleaned_9.csv")
clean_810_proximal = load_csv("C:/Users/k20113376/Documents/Clinical Trial/NEW_Code_Data/Data/Data_Cleaned/Cleaned_data_V4/IMPROVED_V4_line_threshold_0.0025_threshold_1.5_PROXIMAL_810_nicp_data_cleaned_9.csv")
#clean_810_subtracted = load_csv("C:/Users/k20113376/Documents/Clinical Trial/NEW_Code_Data/Data/Data_Cleaned/Cleaned_data_V4/IMPROVED_V4_line_threshold_0.0025_threshold_1.5_SUBTRACTED_810_nicp_data_cleaned_9.csv")

#run_test(clean_810_distal, 60, 100, 100)

# 2,30,31,11
clean_IICP = clean_IICP.drop(columns = ['2','30','31','11'], axis=1)
clean_810_distal = clean_810_distal.drop(columns = ['2','30','31','11'], axis=1)
clean_810_proximal = clean_810_proximal.drop(columns = ['2','30','31','11'], axis=1)

# remove values function using remove_values function
clean_IICP, clean_810_distal, clean_810_proximal = remove_values(clean_IICP, clean_810_distal, clean_810_proximal, upper=60, lower=0)
run_feature_extraction(6000, clean_810_distal, clean_810_proximal, clean_IICP, "Save_29_11_22", debug = 0)
