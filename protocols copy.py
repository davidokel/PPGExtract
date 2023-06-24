import math
from feature_extraction_code.sqis import get_sqis
from support_code.pulse_detection import get_pulses
import numpy as np
import pickle
from feature_extraction_code.features import get_features_v2
import datetime
import pandas as pd

def extraction_protocol_v2(dataset, data, fs, window_size, save_name, visualise = False, debug = False, z_score_threshold = 3, z_score_detection = True, derivative=[]):
    # Calculate the total number of windows
    num_windows = math.ceil(len(data)/window_size)

    aggregate_features = {}
    peaks_list = np.zeros(len(data), dtype='i')
    troughs_list = np.zeros(len(data), dtype='i')

    # Check if len(data) < window_size
    if len(data) < window_size:
        window_size = len(data)

    num_windows = math.ceil(len(data)/window_size) # Calculating the number of windows the data will be split into
    last_window_size = len(data) % window_size # Calculating the size of the last window

    for window in range(num_windows):
        start = window*window_size # Calculating the start index of the current window
        end = start + window_size # Calculating the end index of the current window
        
        if window == num_windows - 1 and last_window_size > 0:
            end = start + window_size + last_window_size # Calculating the end index of the last window

         # +---------------------+   
        # | DEFINING DATA CHUNK |
        # +---------------------+
        data_window = data[start:end] # Defining the data window

        # Print the current window
        print("----------------------------------------------------")
        print(str(window) + "/" + str(num_windows))
        # Print the start and end of the window
        print("Start: " + str(start) + " End: " + str(end))
        # Print the total length of the data without nan values
        print("Length of data: " + str(len(data.dropna())))
        print("----------------------------------------------------")
        print("")
        
        #################################################
        # Extracting the pulses from the current window #
        #################################################
        peak_points, peaks, troughs = get_pulses(list(data_window), fs=fs, visualise=visualise, debug=debug, z_score_threshold=z_score_threshold, z_score_detection=z_score_detection)
        features = get_features_v2(peak_points, fs, visualise=visualise, debug=debug)
        aggregate_features[window] = features
        
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
            aggregate_features[window].update(deriv_dictionary)
        
        print(aggregate_features)
        
        ########################
        # PEAK AND TROUGH LIST #
        ########################
        # Convert the peaks and troughs to numpy arrays
        peaks = np.array(peaks, dtype='i')
        troughs = np.array(troughs, dtype='i')

        # Add the start of the window to the peaks and troughs
        peaks += start
        troughs += start

        peaks_list[peaks] = 1
        troughs_list[troughs] = 1
    
    # Get the date and time and use to save the data   
    # Get the current date and month
    date = datetime.datetime.now().strftime("%d_%m_%Y")
    print(date)

    # Convert the aggregate features to a dataframe
    agg_features = pd.DataFrame.from_dict(aggregate_features, orient='index')
    # Save the aggregate features to a csv file
    agg_features.to_csv("extracted_features/AGG_WINDOW_FEATURES_"+save_name+".csv")

    dataset["Peaks"] = peaks_list
    dataset["Troughs"] = troughs_list
    dataset.to_csv(save_name+"_"+date+".csv")


            
