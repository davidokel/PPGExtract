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

#['2','30','31','11']
#clean_IICP = load_csv("C:/Users/k20113376/Documents/Clinical Trial/NEW_Code_Data/Data/Data_Cleaned/Cleaned_data_V4/IMPROVED_V4_line_threshold_0.0025_threshold_1.5_iicp_data_cleaned_9.csv")
#clean_810_distal = load_csv("C:/Users/k20113376/Documents/Clinical Trial/NEW_Code_Data/Data/Data_Cleaned/Cleaned_data_V4/IMPROVED_V4_line_threshold_0.0025_threshold_1.5_DISTAL_810_nicp_data_cleaned_9.csv")
#clean_810_proximal = load_csv("C:/Users/k20113376/Documents/Clinical Trial/NEW_Code_Data/Data/Data_Cleaned/Cleaned_data_V4/IMPROVED_V4_line_threshold_0.0025_threshold_1.5_PROXIMAL_810_nicp_data_cleaned_9.csv")

clean_IICP = pd.read_csv("C:/Users/k20113376/Documents/Clinical Trial/NEW_Code_Data/Data/Data_Cleaned/Data_All_Wav_16_12_22/Joint Data/IICP/Joint_Data_IICP_16_12_22.csv")
clean_IICP = clean_IICP.drop(columns = ['19','9'], axis=1)

"""# 770 wavelength
clean_770_distal = pd.read_csv("C:/Users/k20113376/Documents/Clinical Trial/NEW_Code_Data/Data/Data_Cleaned/Data_All_Wav_16_12_22/Joint Data/NICP/DIS_770_Joint_Data_IICP_16_12_22.csv")
clean_770_proximal = pd.read_csv("C:/Users/k20113376/Documents/Clinical Trial/NEW_Code_Data/Data/Data_Cleaned/Data_All_Wav_16_12_22/Joint Data/NICP/PROX_770_Joint_Data_IICP_16_12_22.csv")

clean_770_distal = clean_770_distal.drop(columns = ['19','9'], axis=1)
clean_770_proximal = clean_770_proximal.drop(columns = ['19','9'], axis=1)

clean_IICP, clean_770_distal, clean_770_proximal = remove_values(clean_IICP, clean_770_distal, clean_770_proximal, upper=60, lower=0)
run_feature_extraction(6000, clean_770_distal, clean_770_proximal, clean_IICP, "Features_All_Wav/770", debug = 0)"""

"""# 810 wavelength
clean_810_distal = pd.read_csv("C:/Users/k20113376/Documents/Clinical Trial/NEW_Code_Data/Data/Data_Cleaned/Data_All_Wav_16_12_22/Joint Data/NICP/DIS_810_Joint_Data_IICP_16_12_22.csv")
clean_810_proximal = pd.read_csv("C:/Users/k20113376/Documents/Clinical Trial/NEW_Code_Data/Data/Data_Cleaned/Data_All_Wav_16_12_22/Joint Data/NICP/PROX_810_Joint_Data_IICP_16_12_22.csv")

clean_810_distal = clean_810_distal.drop(columns = ['19','9'], axis=1)
clean_810_proximal = clean_810_proximal.drop(columns = ['19','9'], axis=1)

clean_IICP, clean_810_distal, clean_810_proximal = remove_values(clean_IICP, clean_810_distal, clean_810_proximal, upper=60, lower=0)
run_feature_extraction(6000, clean_810_distal, clean_810_proximal, clean_IICP, "Features_All_Wav/810", debug = 0)

# 850 wavelength
clean_850_distal = pd.read_csv("C:/Users/k20113376/Documents/Clinical Trial/NEW_Code_Data/Data/Data_Cleaned/Data_All_Wav_16_12_22/Joint Data/NICP/DIS_850_Joint_Data_IICP_16_12_22.csv")
clean_850_proximal = pd.read_csv("C:/Users/k20113376/Documents/Clinical Trial/NEW_Code_Data/Data/Data_Cleaned/Data_All_Wav_16_12_22/Joint Data/NICP/PROX_850_Joint_Data_IICP_16_12_22.csv")

clean_850_distal = clean_850_distal.drop(columns = ['19','9'], axis=1)
clean_850_proximal = clean_850_proximal.drop(columns = ['19','9'], axis=1)

clean_IICP, clean_850_distal, clean_850_proximal = remove_values(clean_IICP, clean_850_distal, clean_850_proximal, upper=60, lower=0)
run_feature_extraction(6000, clean_850_distal, clean_850_proximal, clean_IICP, "Features_All_Wav/850", debug = 0)

# 880 wavelength
clean_880_distal = pd.read_csv("C:/Users/k20113376/Documents/Clinical Trial/NEW_Code_Data/Data/Data_Cleaned/Data_All_Wav_16_12_22/Joint Data/NICP/DIS_880_Joint_Data_IICP_16_12_22.csv")
clean_880_proximal = pd.read_csv("C:/Users/k20113376/Documents/Clinical Trial/NEW_Code_Data/Data/Data_Cleaned/Data_All_Wav_16_12_22/Joint Data/NICP/PROX_880_Joint_Data_IICP_16_12_22.csv")

clean_880_distal = clean_880_distal.drop(columns = ['19','9'], axis=1)
clean_880_proximal = clean_880_proximal.drop(columns = ['19','9'], axis=1)

clean_IICP, clean_880_distal, clean_880_proximal = remove_values(clean_IICP, clean_880_distal, clean_880_proximal, upper=60, lower=0)
run_feature_extraction(6000, clean_880_distal, clean_880_proximal, clean_IICP, "Features_All_Wav/880", debug = 0)
"""

