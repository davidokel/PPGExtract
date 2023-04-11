import math
from feature_extraction_code.sqis import get_sqis
from feature_extraction_code.data_methods import get_onsets_v2
import numpy as np
import pickle
from feature_extraction_code.features import get_features
from signal_quality_classifiers.dictionary_formatter import dict_to_df
from signal_quality_classifiers.classify_pulses import get_pulse_predictions

def get_data_sqis(dataset, fs, window_size, save_name, visualise = 0, debug = 0):
    # Isolating the data column
    data = dataset["Data"]

    # Calculate the total number of windows
    num_windows = math.ceil(len(data)/window_size)

    pulses = {}
    peaks_list = np.zeros(len(data), dtype='i')
    troughs_list = np.zeros(len(data), dtype='i')

    for window in range(num_windows):
        print(str(window) + "/" + str(num_windows))
        start = window * window_size
        end = start + window_size

        if end > len(data):
            end = len(data)

        peak_points, peaks, troughs = get_onsets_v2(list(data[start:end]), fs, visualise = visualise, debug = debug)
        peak_points = {key + start: value for key, value in peak_points.items()}
        for key in peak_points:
            peak_points[key]["Peak"] += start
            peak_points[key]["Pre_peak"] += start
            peak_points[key]["Post_peak"] += start
        
        peak_points = get_sqis(peak_points, 100, debug=debug)
        pulses.update(peak_points)
        peak_points = get_features(peak_points, visualise=visualise)
        pulses.update(peak_points)
        pulses = get_pulse_predictions(pulses, "signal_quality_classifiers/random_forest_classifier.pkl")

        # convert peaks and troughs to numpy arrays
        peaks = np.array(peaks, dtype='i')
        troughs = np.array(troughs, dtype='i')

        # add start value to each element using numpy broadcasting
        peaks += start
        troughs += start

        peaks_list[peaks] = 1
        troughs_list[troughs] = 1

    with open('DICTIONARY_UNANNOTATED_'+ save_name + '.pkl', 'wb') as f:
        pickle.dump(pulses, f)

    dataset["Peaks"] = peaks_list
    dataset["Troughs"] = troughs_list
    dataset.to_csv(save_name+".csv")



            
