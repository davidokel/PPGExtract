import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp
import math

def get_widths(pulse_data, visualise=0):
    data = pulse_data["raw_pulse_data"]
    pulse_start = 0
    pulse_end = len(data)-1
    peak = np.array([pulse_data["Relative_peak"]])

    # Forcing positive values by adding 1
    data = 1 + data

    # Initialising lists for the half widths and pulse widths
    systolic_widths, pulse_widths = [], []

    # Checking if there are any peaks in the data
    if peak:
        # Calculate the prominence of the peak
        peak_prominence = sp.peak_prominences(data, peak)[0][0]
        contour_heights = data[peak] - peak_prominence # Calculating the contour heights (Used for plotting)

        # Calculating the pulse widths and ratios at 10%, 25% 33%, 50%, 66% and 75% of the peak height
        pulse_widths_res_10 = sp.peak_widths(data, peak, rel_height=0.1)
        pulse_widths_res_25 = sp.peak_widths(data, peak, rel_height=0.25)
        pulse_widths_res_33 = sp.peak_widths(data, peak, rel_height=0.33)
        pulse_widths_res_50 = sp.peak_widths(data, peak, rel_height=0.5)
        pulse_widths_res_66 = sp.peak_widths(data, peak, rel_height=0.66)
        pulse_widths_res_75 = sp.peak_widths(data, peak, rel_height=0.75)

        # Calculating the pulse width by subtracting the end of the pulse from the start of the pulse
        pulse_width = pulse_end - pulse_start

        # Get the index of the peak
        peak_index = int(pulse_widths_res_10[0][0])

        # Get the start and end indices of the pulse width
        width_start_index_10 = pulse_widths_res_10[2][0]
        width_end_index_10 = pulse_widths_res_10[3][0]

        width_start_index_25 = pulse_widths_res_25[2][0]
        width_end_index_25 = pulse_widths_res_25[3][0]

        width_start_index_33 = pulse_widths_res_33[2][0]
        width_end_index_33 = pulse_widths_res_33[3][0]

        width_start_index_50 = pulse_widths_res_50[2][0]
        width_end_index_50 = pulse_widths_res_50[3][0]

        width_start_index_66 = pulse_widths_res_66[2][0]
        width_end_index_66 = pulse_widths_res_66[3][0]

        width_start_index_75 = pulse_widths_res_75[2][0]
        width_end_index_75 = pulse_widths_res_75[3][0]

        # Calcualting the systolic width and diastolic width
        systolic_width_10 = peak_index - width_start_index_10
        diastolic_width_10 = width_end_index_10 - peak_index

        systolic_width_25 = peak_index - width_start_index_25
        diastolic_width_25 = width_end_index_25 - peak_index

        systolic_width_33 = peak_index - width_start_index_33
        diastolic_width_33 = width_end_index_33 - peak_index

        systolic_width_50 = peak_index - width_start_index_50
        diastolic_width_50 = width_end_index_50 - peak_index

        systolic_width_66 = peak_index - width_start_index_66
        diastolic_width_66 = width_end_index_66 - peak_index

        systolic_width_75 = peak_index - width_start_index_75
        diastolic_width_75 = width_end_index_75 - peak_index

        # Calculating the ratio between the diastolic and systolic widths
        ds_ratio_10 = diastolic_width_10 / systolic_width_10
        ds_ratio_25 = diastolic_width_25 / systolic_width_25
        ds_ratio_33 = diastolic_width_33 / systolic_width_33
        ds_ratio_50 = diastolic_width_50 / systolic_width_50
        ds_ratio_66 = diastolic_width_66 / systolic_width_66
        ds_ratio_75 = diastolic_width_75 / systolic_width_75
        
        # Visualising the extracted features
        if visualise == 1:
            fig = plt.figure()
            plt.title("Half peak width and pulse width")
            plt.plot(data)
            plt.hlines(*pulse_widths_res_10[1:], color="C1", label="Peak width (10% Prominence)")
            plt.hlines(*pulse_widths_res_25[1:], color="C2", label="Peak width (25% Prominence)")
            plt.hlines(*pulse_widths_res_33[1:], color="C3", label="Peak width (33% Prominence)")
            plt.hlines(*pulse_widths_res_50[1:], color="C4", label="Peak width (50% Prominence)")
            plt.hlines(*pulse_widths_res_66[1:], color="C5", label="Peak width (66% Prominence)")
            plt.hlines(*pulse_widths_res_75[1:], color="C6", label="Peak width (75% Prominence)")

            # Plot the prominence of the peak
            plt.vlines(x=peak, ymin=contour_heights, ymax=data[peak], label="amplitudes", linestyles="dashed", colors="green")
            
            # Determine if pulse_start or pulse_end has a lower value
            if data[pulse_start] < data[pulse_end]:
                plt.hlines(data[pulse_start], pulse_start, pulse_end, color="C7", label="pulse width")
            else:
                plt.hlines(data[pulse_end], pulse_end, pulse_start, color="C7", label="pulse width")

            plt.legend(loc="upper left")
            plt.show()
    
    else:
        return np.NaN, np.NaN
    