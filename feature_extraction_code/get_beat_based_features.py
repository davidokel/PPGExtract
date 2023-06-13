import numpy as np

def get_beat_features(window_pulse_data, visualise=0):
    peaks = [window_pulse_data[key]["Peak"] for key in window_pulse_data]

    # Calculate the number of pulses per window
    num_pulses = len(peaks)

    # Calculate the interbeat interval
    ibi = np.diff(peaks)

    # Calculate the mean interbeat interval
    mean_ibi = np.mean(ibi)

    # Calculate the standard deviation of the interbeat interval
    std_ibi = np.std(ibi)

    # Calculate the coefficient of variation of the interbeat interval
    cv_ibi = std_ibi/mean_ibi
    
    