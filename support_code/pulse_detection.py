import matplotlib.pyplot as plt
from support_code.data_methods import normalise_data
import numpy as np
import scipy.signal as sp
from scipy import stats
from scipy.stats import norm

def get_pulses(data, fs=100, visualise=False, debug=False):
    data = np.array(data)
    # Normalise the data
    normalised_data = normalise_data(data, fs)
    
    # Filtering the data using a 3rd order Butterworth lowpass filter with a cutoff frequency of 3Hz
    sos_ac = sp.butter(2, 3, btype='lowpass', analog=False, output='sos', fs=fs)
    try:
        data = sp.sosfiltfilt(sos_ac, data, axis= -1, padtype='odd', padlen=None)
    except ValueError:
        min_padlen = int((len(data) - 1) // 2)
        data = sp.sosfiltfilt(sos_ac, data, axis= -1, padtype='odd', padlen=min_padlen)
        
    moving_average_data = np.convolve(data, np.ones((int(fs*0.95),))/int(fs*0.95), mode='valid')
    n = len(data) - len(moving_average_data)
    data = data[int(n/2):-int(n/2)]

    # Find indexes where the moving average and the data intersect
    if len(moving_average_data) >= len(data):
        diff = moving_average_data - data
    elif len(data) <= len(moving_average_data):
        diff = data - moving_average_data 

    crossings = np.where(np.gradient(np.sign(diff)))[0]

    # Find consecutive runs of indices
    differences = np.diff(crossings)
    split_indices = np.where(differences > 10)[0] + 1
    groups = np.split(crossings, split_indices)
    crossings = [group[0] for group in groups if len(group) > 0]

    p_t_tuples = []
    # Saving elements into the p_t_tuples list as (index, "Peak") or (index, "Trough)

    # For every key in data_points, there should be two values, i) "index" the peak or trough index and ii) "P_or_T" if the point is a peak or a trough.
    # Iterate over the crossings in pairs and check if the data between the crossings is above the moving average or not
    # If the data is above the moving average then the get the argmax of the data between the crossings
    # If the data is below the moving average then the get the argmin of the data between the crossings
    for i, j in zip(crossings, crossings[1:]):
        # Calculate the average moving average value between the two crossings
        average_moving_average = np.mean(moving_average_data[i:j])
        # Determine if the data between the two crossings is above or below the moving average
        data_between = data[i:j]
        mean_data_between = np.mean(data_between)
        if mean_data_between > average_moving_average:
            # Get the argmax of the data between the two crossings
            peak = np.argmax(data_between) + i

            # If the p_t_tuples list is not empty, check if the previous element is a peak
            if len(p_t_tuples) > 0 and p_t_tuples[-1][1] == "Peak":
                previous_element = p_t_tuples[-1]
                previous_peak = previous_element[0]
                previous_peak_value = data[previous_peak]
                current_peak_value = data[peak]

                # If the current peak is higher than the previous peak then remove the previous peak
                if current_peak_value > previous_peak_value:
                    p_t_tuples.pop()
                    # Add the current peak to the data_points dictionary
                    p_t_tuples.append((peak, "Peak"))
            else:
                # Add the current peak to the data_points dictionary
                p_t_tuples.append((peak, "Peak"))
                
        elif mean_data_between < average_moving_average:
            # Get the argmin of the data between the two crossings
            trough = np.argmin(data[i:j]) + i
            # Add the trough to the p_t_tuples list
            p_t_tuples.append((trough, "Trough"))

    # Iterate over the p_t_tuples list, if the first element is a peak not a trough then remove it until the first element is a trough
    while (len(p_t_tuples) > 0) and (p_t_tuples[0][1] == "Peak"):
        p_t_tuples.pop(0)
        
    # If the p_t_tuples list is not empty and the last detected element is a peak not a trough then remove it until the last detected element is a trough
    while (len(p_t_tuples) > 0) and (p_t_tuples[-1][1] == "Peak"):
        p_t_tuples.pop(-1)

    # A anomalous peak can be defined as a whose value falls below either its previous or next trough
    anom_peak = []
    # Iterate over the peaks
    for i in range(len(p_t_tuples)):
        # Check if the current element is a peak
        if p_t_tuples[i][1] == "Peak":
            # Get the previous and next troughs
            previous_trough = p_t_tuples[i-1][0]
            next_trough = p_t_tuples[i+1][0]

            # Check if the peak value is below the previous or next trough
            if data[p_t_tuples[i][0]] < data[previous_trough] or data[p_t_tuples[i][0]] < data[next_trough]:
                anom_peak.append(p_t_tuples[i][0])

    # Remove the anomalous peaks from the p_t_tuples list
    for peak in anom_peak:
        p_t_tuples.remove((peak, "Peak"))

    if debug:
        plt.title("Bad pulse removal")
        # Plot just the peaks and troughs
        plt.plot([x[0] for x in p_t_tuples if x[1] == "Trough"], [data[x[0]] for x in p_t_tuples if x[1] == "Trough"], 'ro', label='Troughs')
        plt.plot([x[0] for x in p_t_tuples if x[1] == "Peak"], [data[x[0]] for x in p_t_tuples if x[1] == "Peak"], 'go', label='Peaks')
        plt.plot(data, label='Raw Data')
        plt.legend()
        plt.show()
        
    # If there are multiple troughs between two peaks:
    # If the majority of the data between the troughs slopes downwards then keep the trough with the lowest value and remove the other trough
    # If the majority of the data between the troughs slopes upwards then keep the trough with the lowest value and remove the other trough
    # If data between the troughs is flat then choose then continue to the next peak
    def filter_troughs(p_t_tuples, data):
        # Get all the peaks within the p_t_tuples list
        peaks = [x[0] for x in p_t_tuples if x[1] == "Peak"]
        # Get all the troughs within the p_t_tuples list
        troughs = [x[0] for x in p_t_tuples if x[1] == "Trough"]

        # Iterate over the peaks in overlapping pairs
        for i, j in zip(peaks, peaks[1:]):
            # Get the troughs between the two peaks
            troughs_between = [trough for trough in troughs if i < trough < j]

            # Check if there are more than one trough between the two peaks
            if len(troughs_between) > 1:
                troughs_to_remove = []

                # Iterate over the troughs between in pairs
                for k, l in zip(troughs_between, troughs_between[1:]):
                    # Get the data between the two troughs
                    data_between = data[k:l]

                    # Ensure data_between has sufficient elements to calculate the gradient
                    if len(data_between) < 2:
                        continue

                    # Determine if the majority of the data between the two troughs slopes upwards or downwards
                    gradient = np.gradient(data_between)

                    # If the majority of the data between the two troughs slopes downwards, remove the trough with the higher value
                    if np.sum(gradient < 0) > len(gradient) / 2:
                        # Keep the trough with the lower value
                        if data[k] < data[l]:
                            troughs_to_remove.append(l)
                        else:
                            troughs_to_remove.append(k)
                    # If the majority of the data between the two troughs slopes upwards, remove the trough with the higher value
                    elif np.sum(gradient > 0) > len(gradient) / 2:
                        # Keep the trough with the lower value
                        if data[k] < data[l]:
                            troughs_to_remove.append(l)
                        else:
                            troughs_to_remove.append(k)

                # Remove the troughs between the two peaks from the p_t_tuples list
                for trough in troughs_to_remove:
                    p_t_tuples = [(index, t) for (index, t) in p_t_tuples if not (index == trough and t == "Trough")]

        return p_t_tuples
        
    # Filter the troughs in the p_t_tuples list
    p_t_tuples = filter_troughs(p_t_tuples, data)

    if debug:
        plt.title("Trough filtering")
        # Plot just the peaks and troughs
        plt.plot([x[0] for x in p_t_tuples if x[1] == "Trough"], [data[x[0]] for x in p_t_tuples if x[1] == "Trough"], 'ro', label='Troughs')
        plt.plot([x[0] for x in p_t_tuples if x[1] == "Peak"], [data[x[0]] for x in p_t_tuples if x[1] == "Peak"], 'go', label='Peaks')
        plt.plot(data, label='Raw Data')
        plt.legend()
        plt.show()
    
    # Get peaks and troughs from the p_t_tuples list
    peaks = [x[0] for x in p_t_tuples if x[1] == "Peak"]
    troughs = [x[0] for x in p_t_tuples if x[1] == "Trough"]

    # Remove the anomalous peaks from the peak_points dictionary
    peak_points = {}

    # Remove peak duplicates
    peaks = list(set(peaks))

    # Iterate over the peaks
    for i in range(len(peaks)):
        # Find the two associated troughs for the peak
        left_troughs = [trough for trough in troughs if trough < peaks[i]] # Get all troughs to the left of the current peak
        right_troughs = [trough for trough in troughs if trough > peaks[i]] # Get all troughs to the right of the current peak
        # Get the max left trough
        pulse_onset = left_troughs[-1]
        # Get the min right trough
        pulse_end = right_troughs[0]

        # They key of the peak is the index of the peak
        key = peaks[i]
        peak_points[key] = {"Peak": peaks[i], "Relative_peak": peaks[i] - pulse_onset, "raw_pulse_data": data[pulse_onset:pulse_end], "norm_pulse_data": normalised_data[pulse_onset:pulse_end], "Pre_peak": pulse_onset, "Post_peak": pulse_end}

    ##############################################################
    # Calculating simple quality metrics for the detected pulses #
    ##############################################################
    # Calculate the average distance between the peaks
    peak_distances = np.gradient(peaks)

    # Calculate the value change between the peak and its associated troughs
    onset_peak_y, peak_end_y = [], []
    # Iterate over the peaks
    for i in range(len(peaks)):
        # Get the associated troughs
        left_troughs = [trough for trough in troughs if trough < peaks[i]]
        right_troughs = [trough for trough in troughs if trough > peaks[i]]
        # Get the max left trough
        pulse_onset = left_troughs[-1]
        # Get the min right trough
        pulse_end = right_troughs[0]

        # Get the value change between the peak and its associated troughs
        onset_peak_y.append(data[peaks[i]] - data[pulse_onset])
        peak_end_y.append(data[peaks[i]] - data[pulse_end])

    def z_score(feature_data, threshold=2.5):
        mean, std = np.mean(feature_data), np.std(feature_data)
        z_score = np.abs((feature_data - mean) / std)
        good = z_score < threshold
        return good

    # Iterate over the peaks
    for i in range(len(peaks)):
        # For each peak, calculate the z-score of the peak distance and the value change between the peak and its associated troughs
        peak_distances_z_score = z_score(peak_distances)
        onset_peak_y_z_score = z_score(onset_peak_y)
        peak_end_y_z_score = z_score(peak_end_y)

        # Get the indexes where the z_score is False
        peak_distances_false = np.where(peak_distances_z_score == False)[0]
        onset_peak_y_false = np.where(onset_peak_y_z_score == False)[0]
        peak_end_y_false = np.where(peak_end_y_z_score == False)[0]

        # Combine the indexes
        indexes = np.concatenate((peak_distances_false, onset_peak_y_false, peak_end_y_false))
        # Get the unique indexes
        indexes = np.unique(indexes)

        # Check if the peak exists in the peak_points dictionary before removing it
        for index in indexes:
            peak = peaks[index]
            if peak in peak_points:
                peak_points.pop(peak)

        # Plot the raw data and the moving average
        if visualise:
            # Get the peaks and troughs from the peak_points dictionary
            peaks = [peak_points[peak]["Peak"] for peak in peak_points.keys()]
            # Troughs are pre_peak and post_peak
            troughs = [peak_points[trough]["Pre_peak"] for trough in peak_points.keys()] + [peak_points[trough]["Post_peak"] for trough in peak_points.keys()]

            plt.plot(data, label='Raw Data')
            plt.plot(peaks, data[peaks], 'go', label='Peaks')
            plt.plot(troughs, data[troughs], 'ro', label='Troughs')
            plt.legend()
            plt.show()
        
        return peak_points, peaks, troughs
