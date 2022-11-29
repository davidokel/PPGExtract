import os
import pandas as pd
import math
from amplitudes_widths_prominences import get_amplitudes_widths_prominences
from upslopes_downslopes_rise_times_auc import get_upslopes_downslopes_rise_times_auc
from data_methods import normalise_data
import numpy as np

def run_feature_extraction(window_size, distal_data, proximal_data, icp_data, save_path, debug = 0, visualise = 0):
    # PURPOSE OF THIS FUNCTION: This function segments the distal_data, proximal_data and icp_data into windows of size window_size and extracts the features from each window along with the mean of the ICP data in that window.
    # The features are then saved into either a distal_features or proximal_features csv file.
    #
    # INPUTS:
    # window_size: The size of the window in seconds.
    # distal_data: The distal data.
    # proximal_data: The proximal data.
    # icp_data: The icp data.
    # save_path: The path to the folder where the features will be saved.
    #
    # OUTPUTS:
    # None

    # CREATING FOLDER TO SAVE FEATURES
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # CREATING DATAFRAMES TO STORE FEATURES
    distal_features = pd.DataFrame()
    proximal_features = pd.DataFrame()

    # CREATING A LIST OF THE NAMES OF THE FEATURES
    feature_list = ['Amplitude', 'Half-peak width', 'Upslope', 'Downslope', 'Rise time', 'Decay time', 'AUC', 'Sys AUC', 'Dia AUC', 'AUC Ratio', 'Second Derivative Ratio', 'IICP', 'Patient']
    
    # Check if the columns of the distal, proximal and icp data are the same
    if distal_data.columns.all() == proximal_data.columns.all() == icp_data.columns.all():
        # Loop through each column in the distal, proximal and icp data
        for column in range(len(icp_data.columns)):
            # Check if the current column of the distal_data, proximal_data and icp_data are the same and their length is the same
            if distal_data.columns[column] == proximal_data.columns[column] == icp_data.columns[column] and len(distal_data.iloc[:,column]) == len(proximal_data.iloc[:,column]) == len(icp_data.iloc[:,column]):
                
                # Convert the data to numpy arrays and check if the length of the data is the same and equal to the window_size
                distal_data_np = distal_data.iloc[:,column].dropna().to_numpy()
                proximal_data_np = proximal_data.iloc[:,column].dropna().to_numpy()
                icp_data_np = icp_data.iloc[:,column].dropna().to_numpy()

                num_windows = int(math.floor(len(icp_data_np)/window_size)) # Calculating the number of windows that can be created from the data

                if len(distal_data_np) != 0 and len(proximal_data_np) != 0 and len(icp_data_np) != 0:
                    # Print lengths of data
                    print('Length of distal data: ' + str(len(distal_data_np)))
                    print('Length of proximal data: ' + str(len(proximal_data_np)))
                    print('Length of icp data: ' + str(len(icp_data_np)))

                    print("Extracting features from patient: " + str(icp_data.columns[column]))
                    # Loop through each window
                    chunk_start = 0
                    chunk_end = window_size
                    for window in range(num_windows):
                        if window == 0:
                            print(icp_data_np[0])
                            print(icp_data_np[chunk_start:chunk_end])


                        # Print the current window out of the total number of windows along with the chunk_start and chunk_end
                        print("Window: " + str(window+1) + "/" + str(num_windows) + " Chunk start: " + str(chunk_start) + " Chunk end: " + str(chunk_end))

                        # Isolate the data for the current window
                        distal_data_window = distal_data_np[chunk_start:chunk_end]
                        proximal_data_window = proximal_data_np[chunk_start:chunk_end]
                        icp_data_window = icp_data_np[chunk_start:chunk_end]

                        # Flip the distal_data_window and proximal_data_window
                        distal_data_window = -distal_data_window
                        proximal_data_window = -proximal_data_window

                        # Normalise the distal_data and proximal_data using the normalise_data
                        distal_data_window = normalise_data(distal_data_window, 100)
                        proximal_data_window = normalise_data(proximal_data_window, 100)

                        # Extract the features from the current window of the distal_data and proximal_data using the get_amplitudes_widths_prominences and get_upslopes_downslopes_rise_times_auc functions 
                        # and calculate the mean of the icp_data in that window

                        # Calculate the mean of the icp_data in that window
                        icp_data_mean = np.median(icp_data_window)

                        # Extract the features from the current window of the distal_data and proximal_data using the get_amplitudes_widths_prominences and get_upslopes_downslopes_rise_times_auc functions
                        # iteratively add the features to the distal_features and proximal_features dataframes
                        distal_prominences, distal_half_peak_widths = get_amplitudes_widths_prominences(distal_data_window, fs = 100, visualise=visualise, debug=debug)
                        proximal_prominences, proximal_half_peak_widths = get_amplitudes_widths_prominences(proximal_data_window, fs = 100, visualise=visualise, debug=debug)

                        distal_upslopes, distal_downslopes, distal_rise_times, distal_decay_times, distal_auc, distal_sys_auc, distal_dia_auc, distal_auc_ratios, distal_second_derivative_ratio = get_upslopes_downslopes_rise_times_auc(distal_data_window, fs = 100, visualise=visualise, debug = debug)
                        proximal_upslopes, proximal_downslopes, proximal_rise_times, proximal_decay_times, proximal_auc, proximal_sys_auc, proximal_dia_auc, proximal_auc_ratios, proximal_second_derivative_ratio = get_upslopes_downslopes_rise_times_auc(proximal_data_window, fs = 100, visualise=visualise, debug = debug)

                        # Calculate the current patient number
                        patient_number = icp_data.columns[column]

                        # Iteratively save the extracted features as a row to the distal_features and proximal_features dataframes using pd.concat
                        distal_features = pd.concat([distal_features, pd.DataFrame([[distal_prominences, distal_half_peak_widths, distal_upslopes, distal_downslopes, distal_rise_times, distal_decay_times, distal_auc, distal_sys_auc, distal_dia_auc, distal_auc_ratios, distal_second_derivative_ratio, icp_data_mean, patient_number]], columns = feature_list)], ignore_index = True)
                        proximal_features = pd.concat([proximal_features, pd.DataFrame([[proximal_prominences, proximal_half_peak_widths, proximal_upslopes, proximal_downslopes, proximal_rise_times, proximal_decay_times, proximal_auc, proximal_sys_auc, proximal_dia_auc, proximal_auc_ratios, proximal_second_derivative_ratio, icp_data_mean, patient_number]], columns = feature_list)], ignore_index = True)

                        chunk_start += window_size
                        chunk_end += window_size
                else:
                    print("The length of the data is not the same for all patients")
            else:
                print("The distal, proximal and icp data are not the same")
        
        # Save the distal_features and proximal_features dataframes as csv files
        distal_features.to_csv(save_path+"/distal_features.csv", index = False)
        proximal_features.to_csv(save_path+"/proximal_features.csv", index = False)
    else: 
        print("Columns are not the same")





