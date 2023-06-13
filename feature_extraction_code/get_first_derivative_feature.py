import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp

def get_first_derivative_features(window_pulse_data, visualise=0):
    # Get the pulse data
    pulse_data = window_pulse_data["raw_pulse_data"]
    pulse_start = 0
    pulse_end = len(pulse_data) - 1
    peak = np.array([window_pulse_data["Relative_peak"]])
    peaks = [window_pulse_data[key]["Peak"] for key in window_pulse_data]

    # Calculate the prominences of the peaks
    prominences = sp.peak_prominences(pulse_data, peaks)[0]

    # Calculate the rise time of the peaks as the difference between the peak and the start of the pulse
    rise_times = peak - pulse_start

    # Calculate the fall time of the peaks as the difference between the end of the pulse and the peak
    fall_times = pulse_end - peak

    # Calculate the ratio between the rise time and decay time
    rise_fall_ratio = rise_times/fall_times


    