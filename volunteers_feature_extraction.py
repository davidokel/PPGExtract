import os
import pandas as pd
import math
from amplitudes_widths_prominences import get_amplitudes_widths_prominences
from upslopes_downslopes_rise_times_auc import get_upslopes_downslopes_rise_times_auc
from data_methods import normalise_data
import numpy as np
import matplotlib.pyplot as plt
from os import listdir, mkdir
from os.path import isfile, join 
import re

def run_feature_extraction(window_size,data_path,save_path,debug = 0,visualise = 0):
    # PURPOSE OF THIS FUNCTION: This function segments the distal_data, proximal_data and icp_data into windows of size window_size and extracts the features from each window along with the mean of the ICP data in that window.
    # The features are then saved into either a distal_features or proximal_features csv file.
    #
    # INPUTS:
    # window_size: The size of the window in seconds.
    # icp_data: The icp data.
    # save_path: The path to the folder where the features will be saved.
    #
    # OUTPUTS:
    # None

    # CREATING FOLDER TO SAVE FEATURES
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    files = [file for file in listdir(data_path) if isfile(join(data_path, file))]

    # CREATING DATAFRAMES TO STORE FEATURES
    features = pd.DataFrame()

    for file in files:
        patient_number = re.findall(r'\d+', file)[0]

        loaded_data = pd.read_csv(data_path+file)

        # CREATING A LIST OF THE NAMES OF THE FEATURES
        feature_list = ['Amplitude', 'Half-peak width', 'Upslope', 'Downslope', 'Rise time', 'Decay time', 'AUC', 'Sys AUC', 'Dia AUC', 'AUC Ratio', 'Second Derivative Ratio', 'Position', 'Patient']

        columns = loaded_data.columns

        for column in columns:
            print(column)

            # Convert the data to numpy arrays and check if the length of the data is the same and equal to the window_size
            data_np = loaded_data[column].dropna().to_numpy()
            num_windows = int(math.floor(len(data_np)/window_size)) # Calculating the number of windows that can be created from the data
            
            # Normalise the distal_data and proximal_data using the normalise_data
            data_np = normalise_data(data_np, 100)

            # Loop through each window
            chunk_start = 0
            chunk_end = window_size
            for window in range(num_windows):
                # Print the current window out of the total number of windows along with the chunk_start and chunk_end
                print("Window: " + str(window+1) + "/" + str(num_windows) + " Chunk start: " + str(chunk_start) + " Chunk end: " + str(chunk_end))

                # Isolate the data for the current window
                data_window = data_np[chunk_start:chunk_end]
            
                # Extract the features from the current window of the distal_data and proximal_data using the get_amplitudes_widths_prominences and get_upslopes_downslopes_rise_times_auc functions
                # iteratively add the features to the distal_features and proximal_features dataframes
                prominences, half_peak_widths = get_amplitudes_widths_prominences(data_window, fs = 100, visualise=visualise, debug=debug)
                upslopes, downslopes, rise_times, decay_times, auc, sys_auc, dia_auc, auc_ratios, second_derivative_ratio = get_upslopes_downslopes_rise_times_auc(data_window, fs = 100, visualise=visualise, debug = debug)

                # Iteratively save the extracted features as a row to the distal_features and proximal_features dataframes using pd.concat
                features = pd.concat([features, pd.DataFrame([[prominences, half_peak_widths, upslopes, downslopes, rise_times, decay_times, auc, sys_auc, dia_auc, auc_ratios, second_derivative_ratio, column, patient_number]], columns = feature_list)], ignore_index = True)

                chunk_start += window_size
                chunk_end += window_size

    # Save the features and proximal_features dataframes as csv files
    features.to_csv(save_path+"/features.csv", index = False)

one_min = 6000
fifteen_secs = 1500
five_secs = 500

#run_feature_extraction(five_secs,"healthy_volunteers_data/valsalva_data/segmented_data_proximal_810/","healthy_volunteers_data/valsalva_data/segmented_data_proximal_810/features/",debug = 0,visualise = 1)
#run_feature_extraction(fifteen_secs,"healthy_volunteers_data/tilting_data/downsampled_data/segmented_data_proximal_810/","healthy_volunteers_data/tilting_data/downsampled_data/segmented_data_proximal_810/features/",debug = 0,visualise = 0)
