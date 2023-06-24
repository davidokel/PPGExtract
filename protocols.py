import math
from support_code.pulse_detection import get_pulses
import numpy as np
from feature_extraction_code.features import get_features_v2
import datetime
import pandas as pd
import pickle as pkl

def extraction_protocol_v2(nicp_dataset, icp_dataset, fs, window_size, visualise = False, debug = False, z_score_threshold = 3, z_score_detection = True, derivative=[]):
    # Check if window_size is 0 or negative
    if window_size <= 0:
        raise ValueError("window_size must be greater than 0")
    
    patient_features_df = pd.DataFrame()
    joint_patient_features = pd.DataFrame()

    # Iterate over the columns of the dataset
    for column in nicp_dataset.columns:

        # Check IF column is in the icp dataset raise error
        if column not in icp_dataset.columns:
            raise ValueError("Column " + column + " is not in the icp dataset")

        # Check if the column exists in both datasets
        nicp_data = nicp_dataset[column]
        icp_data = icp_dataset[column]
        # Drop the nan values
        nicp_data = nicp_data.dropna()
        icp_data = icp_data.dropna()

        # Get 10 first rows of the data
        nicp_data = nicp_data
        icp_data = icp_data

        # Check if the length of the data is the same
        if len(nicp_data) != len(icp_data):
            raise ValueError("Length ICP and NICP data is not the same: " + str(len(nicp_data)) + " vs " + str(len(icp_data)) + " for column " + column)
        
        # Calculate the total number of windows
        num_windows = math.ceil(len(nicp_data)/window_size)

        patient_features = {}

        # Check if len(data) < window_size and not 0
        if len(nicp_data) < window_size and len(nicp_data) > 0:
            window_size = len(nicp_data)

        num_windows = math.ceil(len(nicp_data)/window_size) # Calculating the number of windows the data will be split into
        last_window_size = len(nicp_data) % window_size # Calculating the size of the last window

        for window in range(num_windows):
            start = window*window_size # Calculating the start index of the current window
            end = start + window_size # Calculating the end index of the current window
            
            if window == num_windows - 1 and last_window_size > 0:
                end = start + window_size + last_window_size # Calculating the end index of the last window

            # +---------------------+   
            # | DEFINING DATA CHUNK |
            # +---------------------+
            data_window = nicp_data[start:end] # Defining the data window
            icp_window = icp_data[start:end] # Defining the icp window

            # Print the current window
            print("----------------------------------------------------")
            print(str(window) + "/" + str(num_windows))
            print("Column: " + str(column))
            # Print the start and end of the window
            print("Start: " + str(start) + " End: " + str(end))
            # Print the total length of the data without nan values
            print("Length of data: " + str(len(nicp_data.dropna())))
            print("----------------------------------------------------")
            print("")
            
            #################################################
            # Extracting the pulses from the current window #
            #################################################
            peak_points, peaks, troughs = get_pulses(list(data_window), fs=fs, visualise=visualise, debug=debug, z_score_threshold=z_score_threshold, z_score_detection=z_score_detection)
            features = get_features_v2(peak_points, fs, visualise=visualise, debug=debug)
            patient_features[window] = features
            
            # Check if derivative is not empty
            if derivative:
                deriv_dictionary = {}
                for i in derivative:
                    #################################################
                    # Extracting the pulses from the current window #
                    #################################################
                    peak_points, peaks, troughs = get_pulses(list(data_window), fs=fs, visualise=visualise, debug=debug, z_score_threshold=z_score_threshold, z_score_detection=z_score_detection, derivative=i)
                    features = get_features_v2(peak_points, fs, visualise=visualise, debug=debug)
                    # Change the name of each value in the dictionary to include the derivative
                    features = {key + "_deriv_" + str(i): value for key, value in features.items()}
                    deriv_dictionary.update(features)
                patient_features[window].update(deriv_dictionary)
            
            patient_features[window]["mean_icp"] = np.mean(icp_window)
            patient_features[window]["median_icp"] = np.median(icp_window)
            patient_features[window]["column"] = column

        # Convert the dictionary to a dataframe
        patient_features_df = pd.DataFrame.from_dict(patient_features, orient='index')
        # Save the patient features to a pickle file
        patient_features_df.to_pickle("D:\\Datasets\\Features\\Patient_level\\patient_features_" + column + ".pkl")
        # Add the dataframe to the joint dataframe using the concat function
        joint_patient_features = pd.concat([joint_patient_features, patient_features_df], axis=0)

    # Get the current date
    now = datetime.datetime.now()
    # Save the joint dataframe to a pickle file with the current date
    joint_patient_features.to_pickle("D:\\Datasets\\Features\\joint_patient_features_" + now.strftime("%Y-%m-%d") + ".pkl")




            
