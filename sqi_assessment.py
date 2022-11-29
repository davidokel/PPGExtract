# Import libraries
import random as rd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp
from data_methods import *
from protocol import *
import os
import plotly.graph_objects as go
import scipy.stats as st

def visual_assessment(data,IICP_data,segment_size, number_of_samples, folder_save_path, data_fs = 100):
    # PURPOSE OF THIS FUNCTION:
    # This function is used to visually assess the data.
    # It will plot a random segment_size segment of data from a random patient along with the skew, kurt, snr, zcr, ent, pi of the segment and ask the user if the segment is good, bad or neutral. 
    # The user will then be asked to input a number between 1 and 3, where 1 is good, 2 is bad and 3 is neutral.
    # The skew, kurt, snr, zcr, ent, pi of the segment will be added to the appropriate csv file depending on the user's input along with the data of the segment, the average IICP_data of the segment and patient number.
    # This function will then be called again until the number of samples specified by the user has been reached.

    # INPUTS:
    # data: The data to be assessed.
    # number_of_samples: The number of samples to be assessed.
    # folder_save_path: The path to the folder where the csv files will be saved.
    # data_fs: The sampling frequency of the data.

    # OUTPUTS:
    # None

    # Select a random patient from the data and plot a random segment_size segment of data from the patient.
    # Loop over the number of samples specified by the user.

    # CREATING FOLDER TO SAVE CSV FILES
    if not os.path.exists(folder_save_path):
        os.makedirs(folder_save_path)

    # CREATING CSV FILES
    good_data = pd.DataFrame(columns = ["Patient", "Segment", "Skew", "Kurt", "SNR", "ZCR", "ENT", "PI", "Mean IICP"])
    bad_data = pd.DataFrame(columns = ["Patient", "Segment", "Skew", "Kurt", "SNR", "ZCR", "ENT", "PI", "Mean IICP"])
    neutral_data = pd.DataFrame(columns = ["Patient", "Segment", "Skew", "Kurt", "SNR", "ZCR", "ENT", "PI", "Mean IICP"])

    for i in range(number_of_samples):
        patient = rd.choice(data.columns)
        segment = rd.randint(0, len(data[patient]) - segment_size)

        # If data includes nans, do not plot the segment
        if np.isnan(data[patient][segment:segment + segment_size]).any():
            continue
        else:
            chunk_filtered = band_pass_filter(data[patient][segment:segment + segment_size], 2, 100, 0.5, 12)

            # Plot filtered data using plotly
            # Replace previous tab with new tab
            fig = go.Figure()
            fig.add_trace(go.Scatter(x = np.arange(0, segment_size/data_fs, 1/data_fs), y = chunk_filtered, mode = 'lines', name = 'Filtered Data'))
            fig.update_layout(title = 'Patient ' + patient + ' Segment ' + str(segment), xaxis_title = 'Time (s)', yaxis_title = 'Amplitude (mV)')
            fig.show()
            
            # Using the get_sqis function, get the SQIs of the segment.
            skew, kurt, snr, zcr, ent, pi, _ = get_sqis(data[patient][segment:segment + segment_size], data_fs)

            os.system('cls' if os.name == 'nt' else 'clear')

            # Print the SQIs of the segment.
            print("Skew: " + str(skew))
            print("Kurt: " + str(kurt))
            print("SNR: " + str(snr))
            print("ZCR: " + str(zcr))
            print("ENT: " + str(ent))
            print("PI: " + str(pi))
            print(" ")

            print("###############################################")
            # Ask the user if the segment is good, bad or neutral.
            print("Is this segment good, bad or other?")
            print("1: Good")
            print("2: Bad")
            print("3: Other")
            user_input = input("Enter a number between 1 and 3: ")

            # Add the SQIs of the segment to the appropriate csv file depending on the user's input along with the data of the segment, the patient number and the segment number.
            if user_input == "1":
                good_data = good_data.append({"Patient": patient, "Segment": data[patient][segment:segment + segment_size], "Skew": skew, "Kurt": kurt, "SNR": snr, "ZCR": zcr, "ENT": ent, "PI": pi, "Mean IICP": np.mean(IICP_data[patient][segment:segment + segment_size])}, ignore_index = True)
            elif user_input == "2":
                bad_data = bad_data.append({"Patient": patient, "Segment": data[patient][segment:segment + segment_size], "Skew": skew, "Kurt": kurt, "SNR": snr, "ZCR": zcr, "ENT": ent, "PI": pi, "Mean IICP": np.mean(IICP_data[patient][segment:segment + segment_size])}, ignore_index = True)
            elif user_input == "3":
                neutral_data = neutral_data.append({"Patient": patient, "Segment": data[patient][segment:segment + segment_size], "Skew": skew, "Kurt": kurt, "SNR": snr, "ZCR": zcr, "ENT": ent, "PI": pi, "Mean IICP": np.mean(IICP_data[patient][segment:segment + segment_size])}, ignore_index = True)
            else:
                print("Invalid input. Please try again.")
                i -= 1
                continue

    # Save the csv files.
    good_data.to_csv(folder_save_path + "good_data.csv", index = False)
    bad_data.to_csv(folder_save_path + "bad_data.csv", index = False)
    neutral_data.to_csv(folder_save_path + "neutral_data.csv", index = False)

def calculate_sqi_data(data, folder_save_path):
    # PURPOSE OF FUNCTION:
    # This reads in the csv file and calculates a range of confidence intervals from 75 to 95 in steps of 5 of the Skew, Kurt, SNR, ZCR, ENT and PI columns and saves the confidence intervals to a new csv file with a label of the confidence interval.
    # This process is repeated for confidence intervals calculated using t-distribution and normal distribution.

    # INPUTS:
    # data: The data to be assessed.
    # folder_save_path: The path to the folder where the csv files will be saved.

    # OUTPUTS:
    # None

    # Read in the csv file.
    data = pd.read_csv(data)

    # Create folder
    if not os.path.exists(folder_save_path):
        os.makedirs(folder_save_path)

    # Creating empty lists to store the confidence intervals
    confidence_intervals = []
    skew_confidence_intervals = []
    kurt_confidence_intervals = []
    snr_confidence_intervals = []
    zcr_confidence_intervals = []
    ent_confidence_intervals = []
    pi_confidence_intervals = []

    # Calculating confidence intervals using t-distribution
    for i in range(75, 100, 5):
        confidence_intervals.append(i)
        skew_confidence_intervals.append(st.t.interval(i/100, len(data["Skew"]) - 1, loc = np.mean(data["Skew"]), scale = st.sem(data["Skew"])))
        kurt_confidence_intervals.append(st.t.interval(i/100, len(data["Kurt"]) - 1, loc = np.mean(data["Kurt"]), scale = st.sem(data["Kurt"])))
        snr_confidence_intervals.append(st.t.interval(i/100, len(data["SNR"]) - 1, loc = np.mean(data["SNR"]), scale = st.sem(data["SNR"])))
        zcr_confidence_intervals.append(st.t.interval(i/100, len(data["ZCR"]) - 1, loc = np.mean(data["ZCR"]), scale = st.sem(data["ZCR"])))
        ent_confidence_intervals.append(st.t.interval(i/100, len(data["ENT"]) - 1, loc = np.mean(data["ENT"]), scale = st.sem(data["ENT"])))
        pi_confidence_intervals.append(st.t.interval(i/100, len(data["PI"]) - 1, loc = np.mean(data["PI"]), scale = st.sem(data["PI"])))

    # Create a new csv file with the confidence intervals.
    confidence_intervals = pd.DataFrame({"Confidence Interval": confidence_intervals, "Skew": skew_confidence_intervals, "Kurt": kurt_confidence_intervals, "SNR": snr_confidence_intervals, "ZCR": zcr_confidence_intervals, "ENT": ent_confidence_intervals, "PI": pi_confidence_intervals})
    confidence_intervals.to_csv(folder_save_path + "confidence_intervals_t_distribution.csv", index = False)

    # Creating empty lists to store the confidence intervals
    confidence_intervals = []
    skew_confidence_intervals = []
    kurt_confidence_intervals = []
    snr_confidence_intervals = []
    zcr_confidence_intervals = []
    ent_confidence_intervals = []
    pi_confidence_intervals = []

    # Calculating confidence intervals using normal distribution
    for i in range(75, 100, 5):
        confidence_intervals.append(i)
        skew_confidence_intervals.append(st.norm.interval(i/100, loc = np.mean(data["Skew"]), scale = st.sem(data["Skew"])))
        kurt_confidence_intervals.append(st.norm.interval(i/100, loc = np.mean(data["Kurt"]), scale = st.sem(data["Kurt"])))
        snr_confidence_intervals.append(st.norm.interval(i/100, loc = np.mean(data["SNR"]), scale = st.sem(data["SNR"])))
        zcr_confidence_intervals.append(st.norm.interval(i/100, loc = np.mean(data["ZCR"]), scale = st.sem(data["ZCR"])))
        ent_confidence_intervals.append(st.norm.interval(i/100, loc = np.mean(data["ENT"]), scale = st.sem(data["ENT"])))
        pi_confidence_intervals.append(st.norm.interval(i/100, loc = np.mean(data["PI"]), scale = st.sem(data["PI"])))
    
    # Create a new csv file with the confidence intervals.
    confidence_intervals = pd.DataFrame({"Confidence Interval": confidence_intervals, "Skew": skew_confidence_intervals, "Kurt": kurt_confidence_intervals, "SNR": snr_confidence_intervals, "ZCR": zcr_confidence_intervals, "ENT": ent_confidence_intervals, "PI": pi_confidence_intervals})
    confidence_intervals.to_csv(folder_save_path + "confidence_intervals_normal_distribution.csv", index = False)

#clean_IICP = load_csv("C:/Users/k20113376/Documents/Clinical Trial/NEW_Code_Data/Data/Data_Cleaned/Cleaned_data_V4/IMPROVED_V4_line_threshold_0.0025_threshold_1.5_iicp_data_cleaned_9.csv")
#clean_810_distal = load_csv("C:/Users/k20113376/Documents/Clinical Trial/NEW_Code_Data/Data/Data_Cleaned/Cleaned_data_V4/IMPROVED_V4_line_threshold_0.0025_threshold_1.5_DISTAL_810_nicp_data_cleaned_9.csv")
#clean_810_proximal = load_csv("C:/Users/k20113376/Documents/Clinical Trial/NEW_Code_Data/Data/Data_Cleaned/Cleaned_data_V4/IMPROVED_V4_line_threshold_0.0025_threshold_1.5_PROXIMAL_810_nicp_data_cleaned_9.csv")
#clean_810_subtracted = load_csv("C:/Users/k20113376/Documents/Clinical Trial/NEW_Code_Data/Data/Data_Cleaned/Cleaned_data_V4/IMPROVED_V4_line_threshold_0.0025_threshold_1.5_SUBTRACTED_810_nicp_data_cleaned_9.csv")

# call visual_assessment function
#visual_assessment(clean_810_distal,clean_IICP,6000,500,"Features/Distal/Visual_Assessment/")

# call calculate_sqi_data function
#calculate_sqi_data("Features/Joint_Features/Visual_Assessment_DISTAL/good_data.csv", "Features/Joint_Features/Visual_Assessment_DISTAL/")

