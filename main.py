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
import random

#['2','30','31','11']
#clean_IICP = load_csv("C:/Users/k20113376/Documents/Clinical Trial/NEW_Code_Data/Data/Data_Cleaned/Cleaned_data_V4/IMPROVED_V4_line_threshold_0.0025_threshold_1.5_iicp_data_cleaned_9.csv")
#clean_810_distal = load_csv("C:/Users/k20113376/Documents/Clinical Trial/NEW_Code_Data/Data/Data_Cleaned/Cleaned_data_V4/IMPROVED_V4_line_threshold_0.0025_threshold_1.5_DISTAL_810_nicp_data_cleaned_9.csv")
#clean_810_proximal = load_csv("C:/Users/k20113376/Documents/Clinical Trial/NEW_Code_Data/Data/Data_Cleaned/Cleaned_data_V4/IMPROVED_V4_line_threshold_0.0025_threshold_1.5_PROXIMAL_810_nicp_data_cleaned_9.csv")

"""removal_list = ['2','30','31','11', '19', '9']

clean_IICP = pd.read_csv("C:/Users/k20113376/Documents/Clinical Trial/NEW_Code_Data/Data/Data_Cleaned/Data_All_Wav_16_12_22/Joint Data/IICP/Joint_Data_IICP_16_12_22.csv")
clean_IICP = clean_IICP.drop(columns = ['19','9'], axis=1)"""

"""# 770 wavelength
clean_770_distal = pd.read_csv("C:/Users/k20113376/Documents/Clinical Trial/NEW_Code_Data/Data/Data_Cleaned/Data_All_Wav_16_12_22/Joint Data/NICP/DIS_770_Joint_Data_IICP_16_12_22.csv")
clean_770_proximal = pd.read_csv("C:/Users/k20113376/Documents/Clinical Trial/NEW_Code_Data/Data/Data_Cleaned/Data_All_Wav_16_12_22/Joint Data/NICP/PROX_770_Joint_Data_IICP_16_12_22.csv")

clean_770_distal = clean_770_distal.drop(columns = ['19','9'], axis=1)
clean_770_proximal = clean_770_proximal.drop(columns = ['19','9'], axis=1)

clean_IICP, clean_770_distal, clean_770_proximal = remove_values(clean_IICP, clean_770_distal, clean_770_proximal, upper=60, lower=0)
run_feature_extraction(6000, clean_770_distal, clean_770_proximal, clean_IICP, "26_01_2023/770", debug = 0, visualise = 1)"""

"""# 810 wavelength
clean_810_distal = pd.read_csv("C:/Users/k20113376/Documents/Clinical Trial/NEW_Code_Data/Data/Data_Cleaned/Data_All_Wav_16_12_22/Joint Data/NICP/DIS_810_Joint_Data_IICP_16_12_22.csv")
clean_810_proximal = pd.read_csv("C:/Users/k20113376/Documents/Clinical Trial/NEW_Code_Data/Data/Data_Cleaned/Data_All_Wav_16_12_22/Joint Data/NICP/PROX_810_Joint_Data_IICP_16_12_22.csv")

clean_810_distal = clean_810_distal.drop(columns = ['19','9'], axis=1)
clean_810_proximal = clean_810_proximal.drop(columns = ['19','9'], axis=1)

clean_IICP, clean_810_distal, clean_810_proximal = remove_values(clean_IICP, clean_810_distal, clean_810_proximal, upper=60, lower=0)
run_feature_extraction(6000, clean_810_distal, clean_810_proximal, clean_IICP, "26_01_2023/810", debug = 0, visualise=1)"""

"""# 850 wavelength
clean_850_distal = pd.read_csv("C:/Users/k20113376/Documents/Clinical Trial/NEW_Code_Data/Data/Data_Cleaned/Data_All_Wav_16_12_22/Joint Data/NICP/DIS_850_Joint_Data_IICP_16_12_22.csv")
clean_850_proximal = pd.read_csv("C:/Users/k20113376/Documents/Clinical Trial/NEW_Code_Data/Data/Data_Cleaned/Data_All_Wav_16_12_22/Joint Data/NICP/PROX_850_Joint_Data_IICP_16_12_22.csv")

clean_850_distal = clean_850_distal.drop(columns = ['19','9'], axis=1)
clean_850_proximal = clean_850_proximal.drop(columns = ['19','9'], axis=1)

clean_IICP, clean_850_distal, clean_850_proximal = remove_values(clean_IICP, clean_850_distal, clean_850_proximal, upper=60, lower=0)
run_feature_extraction(6000, clean_850_distal, clean_850_proximal, clean_IICP, "26_01_2023/850", debug = 0, visualise=1)"""

"""# 880 wavelength
clean_880_distal = pd.read_csv("C:/Users/k20113376/Documents/Clinical Trial/NEW_Code_Data/Data/Data_Cleaned/Data_All_Wav_16_12_22/Joint Data/NICP/DIS_880_Joint_Data_IICP_16_12_22.csv")
clean_880_proximal = pd.read_csv("C:/Users/k20113376/Documents/Clinical Trial/NEW_Code_Data/Data/Data_Cleaned/Data_All_Wav_16_12_22/Joint Data/NICP/PROX_880_Joint_Data_IICP_16_12_22.csv")

clean_880_distal = clean_880_distal.drop(columns = ['19','9'], axis=1)
clean_880_proximal = clean_880_proximal.drop(columns = ['19','9'], axis=1)

clean_IICP, clean_880_distal, clean_880_proximal = remove_values(clean_IICP, clean_880_distal, clean_880_proximal, upper=60, lower=0)
run_feature_extraction(6000, clean_880_distal, clean_880_proximal, clean_IICP, "26_01_2023/880", debug = 0, visualise=1)"""

clean_770_distal = pd.read_csv("C:/Users/k20113376/Documents/Clinical Trial/NEW_Code_Data/Data/Data_Cleaned/Data_All_Wav_16_12_22/Joint Data/NICP/DIS_770_Joint_Data_IICP_16_12_22.csv")
clean_880_distal = pd.read_csv("C:/Users/k20113376/Documents/Clinical Trial/NEW_Code_Data/Data/Data_Cleaned/Data_All_Wav_16_12_22/Joint Data/NICP/DIS_880_Joint_Data_IICP_16_12_22.csv")
clean_850_distal = pd.read_csv("C:/Users/k20113376/Documents/Clinical Trial/NEW_Code_Data/Data/Data_Cleaned/Data_All_Wav_16_12_22/Joint Data/NICP/DIS_850_Joint_Data_IICP_16_12_22.csv")
clean_810_distal = pd.read_csv("C:/Users/k20113376/Documents/Clinical Trial/NEW_Code_Data/Data/Data_Cleaned/Data_All_Wav_16_12_22/Joint Data/NICP/DIS_810_Joint_Data_IICP_16_12_22.csv")

clean_770_distal = clean_770_distal.drop(columns = ['19','9'], axis=1)
clean_880_distal = clean_880_distal.drop(columns = ['19','9'], axis=1)
clean_850_distal = clean_850_distal.drop(columns = ['19','9'], axis=1)
clean_810_distal = clean_810_distal.drop(columns = ['19','9'], axis=1)

# Get length of the data
length = len(clean_770_distal)

# Get number of unique values in the "Patient" column
unique_patients = list(clean_770_distal.columns)

# Create list of value from 0 to length in steps of 6000
min = 6000
mins = 10
instances = min * mins
x = np.arange(0, length, instances)

# Iterate over each patient and plot 6000 values

# Iterate over the x values
"""for i in range(0, len(x)):
    # Iterate over each patient
    for j in range(0, len(unique_patients)):
        # Plot the data
        plt.plot(clean_770_distal[unique_patients[j]][x[i]:x[i]+6000], label = "770")
        plt.plot(clean_880_distal[unique_patients[j]][x[i]:x[i]+6000], label = "880")
        plt.plot(clean_850_distal[unique_patients[j]][x[i]:x[i]+6000], label = "850")
        plt.plot(clean_810_distal[unique_patients[j]][x[i]:x[i]+6000], label = "810")
        plt.legend()
        plt.title("Patient: " + str(unique_patients[j]) + " - " + str(x[i]) + " to " + str(x[i]+6000))

        # Save the image as full size
        plt.savefig(str(unique_patients[j]) + "_" + str(x[i]) + "_" + str(x[i]+6000) + ".png", dpi=300)
        plt.show()"""

for i in x:
    start = i
    end = i + instances
    end_zoom = i + int((6000/4))
    # Plot the window
    plt.plot(clean_770_distal['23'][start:end], label = "770")
    plt.plot(clean_880_distal['23'][start:end], label = "880")
    plt.plot(clean_850_distal['23'][start:end], label = "850")
    plt.plot(clean_810_distal['23'][start:end], label = "810")

    # Add an "zoomed in" section of the plot to the right
    ax2 = plt.axes([0.6, 0.6, 0.25, 0.25])
    plt.plot(clean_770_distal['23'][start:end_zoom], label = "770")
    plt.plot(clean_880_distal['23'][start:end_zoom], label = "880")
    plt.plot(clean_850_distal['23'][start:end_zoom], label = "850")
    plt.plot(clean_810_distal['23'][start:end_zoom], label = "810")
    plt.xlim(start, end_zoom)
    plt.ylim(0, 60)
    
    plt.title("Patient: " + str(23) + " (Distal Data)")
    plt.legend()
    plt.show()

    



