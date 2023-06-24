import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp
import math

def get_widths(window_pulse_data, visualise=False, debug=False):

    if window_pulse_data:
        ###########################################################
        # Initialising a dictionary and lists to store WIDTH data #
        ###########################################################
        pulse_widths = []
        pulse_widths_10, pulse_widths_20, pulse_widths_30, pulse_widths_40, pulse_widths_50, pulse_widths_60, pulse_widths_70, pulse_widths_80, pulse_widths_90 = [], [], [], [], [], [], [], [], []
        systolic_widths_10, systolic_widths_20, systolic_widths_30, systolic_widths_40, systolic_widths_50, systolic_widths_60, systolic_widths_70, systolic_widths_80, systolic_widths_90 = [], [], [], [], [], [], [], [], []
        diastolic_widths_10, diastolic_widths_20, diastolic_widths_30, diastolic_widths_40, diastolic_widths_50, diastolic_widths_60, diastolic_widths_70, diastolic_widths_80, diastolic_widths_90 = [], [], [], [], [], [], [], [], []
        ds_ratios_10, ds_ratios_20, ds_ratios_30, ds_ratios_40, ds_ratios_50, ds_ratios_60, ds_ratios_70, ds_ratios_80, ds_ratios_90 = [], [], [], [], [], [], [], [], []

        # Iterate over the keys of the dictionary, every key represents a pulse within the window
        for key in window_pulse_data:

            data = window_pulse_data[key]["pulse_data"]
            pulse_start = 0
            pulse_end = len(data)-1
            peak = np.array([window_pulse_data[key]["relative_peak"]])

            # Checking if there are any peaks in the data
            if peak:
                # Calculate the prominence of the peak
                peak_prominence = sp.peak_prominences(data, peak)[0][0]
                contour_heights = data[peak] - peak_prominence # Calculating the contour heights (Used for plotting)

                # Calculating the pulse width by subtracting the end of the pulse from the start of the pulse
                pulse_width = pulse_end - pulse_start
                pulse_widths.append(pulse_width)

                # Calculating the pulse widths and ratios at 10%,20%,30%,40%,50%,60%,70%,80%,90% of the pulse prominence
                pulse_width_10 = sp.peak_widths(data, peak, rel_height=0.1)
                pulse_width_20 = sp.peak_widths(data, peak, rel_height=0.2)
                pulse_width_30 = sp.peak_widths(data, peak, rel_height=0.3)
                pulse_width_40 = sp.peak_widths(data, peak, rel_height=0.4)
                pulse_width_50 = sp.peak_widths(data, peak, rel_height=0.5)
                pulse_width_60 = sp.peak_widths(data, peak, rel_height=0.6)
                pulse_width_70 = sp.peak_widths(data, peak, rel_height=0.7)
                pulse_width_80 = sp.peak_widths(data, peak, rel_height=0.8)
                pulse_width_90 = sp.peak_widths(data, peak, rel_height=0.9)

                pulse_widths_10.append(pulse_width_10[0][0])
                pulse_widths_20.append(pulse_width_20[0][0])
                pulse_widths_30.append(pulse_width_30[0][0])
                pulse_widths_40.append(pulse_width_40[0][0])
                pulse_widths_50.append(pulse_width_50[0][0])
                pulse_widths_60.append(pulse_width_60[0][0])
                pulse_widths_70.append(pulse_width_70[0][0])
                pulse_widths_80.append(pulse_width_80[0][0])
                pulse_widths_90.append(pulse_width_90[0][0])
                
                # Get the index of the peak
                peak_index = peak[0]

                # Get the start and end indices of the pulse width
                width_start_index_10 = pulse_width_10[2][0]
                width_end_index_10 = pulse_width_10[3][0]

                width_start_index_20 = pulse_width_20[2][0]
                width_end_index_20 = pulse_width_20[3][0]

                width_start_index_30 = pulse_width_30[2][0]
                width_end_index_30 = pulse_width_30[3][0]

                width_start_index_40 = pulse_width_40[2][0]
                width_end_index_40 = pulse_width_40[3][0]

                width_start_index_50 = pulse_width_50[2][0]
                width_end_index_50 = pulse_width_50[3][0]

                width_start_index_60 = pulse_width_60[2][0]
                width_end_index_60 = pulse_width_60[3][0]

                width_start_index_70 = pulse_width_70[2][0]
                width_end_index_70 = pulse_width_70[3][0]

                width_start_index_80 = pulse_width_80[2][0]
                width_end_index_80 = pulse_width_80[3][0]

                width_start_index_90 = pulse_width_90[2][0]
                width_end_index_90 = pulse_width_90[3][0]

                # Calcualting the systolic width and diastolic width
                systolic_width_10 = peak_index - width_start_index_10
                diastolic_width_10 = width_end_index_10 - peak_index
                systolic_widths_10.append(systolic_width_10)
                diastolic_widths_10.append(diastolic_width_10)

                systolic_width_20 = peak_index - width_start_index_20
                diastolic_width_20 = width_end_index_20 - peak_index
                systolic_widths_20.append(systolic_width_20)
                diastolic_widths_20.append(diastolic_width_20)

                systolic_width_30 = peak_index - width_start_index_30
                diastolic_width_30 = width_end_index_30 - peak_index
                systolic_widths_30.append(systolic_width_30)
                diastolic_widths_30.append(diastolic_width_30)
                
                systolic_width_40 = peak_index - width_start_index_40
                diastolic_width_40 = width_end_index_40 - peak_index
                systolic_widths_40.append(systolic_width_40)
                diastolic_widths_40.append(diastolic_width_40)
                
                systolic_width_50 = peak_index - width_start_index_50
                diastolic_width_50 = width_end_index_50 - peak_index
                systolic_widths_50.append(systolic_width_50)
                diastolic_widths_50.append(diastolic_width_50)

                systolic_width_60 = peak_index - width_start_index_60
                diastolic_width_60 = width_end_index_60 - peak_index
                systolic_widths_60.append(systolic_width_60)
                diastolic_widths_60.append(diastolic_width_60)
                
                systolic_width_70 = peak_index - width_start_index_70
                diastolic_width_70 = width_end_index_70 - peak_index
                systolic_widths_70.append(systolic_width_70)
                diastolic_widths_70.append(diastolic_width_70)

                systolic_width_80 = peak_index - width_start_index_80
                diastolic_width_80 = width_end_index_80 - peak_index
                systolic_widths_80.append(systolic_width_80)
                diastolic_widths_80.append(diastolic_width_80)

                systolic_width_90 = peak_index - width_start_index_90
                diastolic_width_90 = width_end_index_90 - peak_index
                systolic_widths_90.append(systolic_width_90)
                diastolic_widths_90.append(diastolic_width_90)

                # Calculating the ratio between the diastolic and systolic widths
                ds_ratio_10 = diastolic_width_10 / systolic_width_10 if systolic_width_10 != 0 else 0
                ds_ratio_20 = diastolic_width_20 / systolic_width_20 if systolic_width_20 != 0 else 0
                ds_ratio_30 = diastolic_width_30 / systolic_width_30 if systolic_width_30 != 0 else 0
                ds_ratio_40 = diastolic_width_40 / systolic_width_40 if systolic_width_40 != 0 else 0
                ds_ratio_50 = diastolic_width_50 / systolic_width_50 if systolic_width_50 != 0 else 0
                ds_ratio_60 = diastolic_width_60 / systolic_width_60 if systolic_width_60 != 0 else 0
                ds_ratio_70 = diastolic_width_70 / systolic_width_70 if systolic_width_70 != 0 else 0
                ds_ratio_80 = diastolic_width_80 / systolic_width_80 if systolic_width_80 != 0 else 0   
                ds_ratio_90 = diastolic_width_90 / systolic_width_90 if systolic_width_90 != 0 else 0

                ds_ratios_10.append(ds_ratio_10)
                ds_ratios_20.append(ds_ratio_20)
                ds_ratios_30.append(ds_ratio_30)
                ds_ratios_40.append(ds_ratio_40)
                ds_ratios_50.append(ds_ratio_50)
                ds_ratios_60.append(ds_ratio_60)
                ds_ratios_70.append(ds_ratio_70)
                ds_ratios_80.append(ds_ratio_80)
                ds_ratios_90.append(ds_ratio_90)
                
                ############
                # Plotting #
                ############
                if visualise:
                    plt.title("Pulse widths")
                    plt.plot(data)
                    plt.hlines(*pulse_width_10[1:], color="C1", label="Pulse width (10% Prominence)")
                    plt.hlines(*pulse_width_20[1:], color="C2", label="Pulse width (20% Prominence)")
                    plt.hlines(*pulse_width_30[1:], color="C3", label="Pulse width (30% Prominence)")
                    plt.hlines(*pulse_width_40[1:], color="C4", label="Pulse width (40% Prominence)")
                    plt.hlines(*pulse_width_50[1:], color="C5", label="Pulse width (50% Prominence)")
                    plt.hlines(*pulse_width_60[1:], color="C6", label="Pulse width (60% Prominence)")
                    plt.hlines(*pulse_width_70[1:], color="C7", label="Pulse width (70% Prominence)")
                    plt.hlines(*pulse_width_80[1:], color="C8", label="Pulse width (80% Prominence)")
                    plt.hlines(*pulse_width_90[1:], color="C9", label="Pulse width (90% Prominence)")
                    
                    # Plot the prominence of the peak
                    plt.vlines(x=peak, ymin=contour_heights, ymax=data[peak], label="Prominence", linestyles="dashed", colors="green")
                    # Plot the peak as a green dot
                    plt.plot(peak, data[peak], "o", color="green", label="Peak")
                    
                    # Determine if pulse_start or pulse_end has a lower value
                    if data[pulse_start] < data[pulse_end]:
                        plt.hlines(data[pulse_start], pulse_start, pulse_end, color="C7", label="pulse width")
                    else:
                        plt.hlines(data[pulse_end], pulse_end, pulse_start, color="C7", label="pulse width")

                    plt.legend(loc="upper left")
                    plt.show()

                    # Get the user input to see if they want to move onto the next visualisation or stop visualising
                    user_input = input("Press enter to continue, or type 'stop' to stop visualising: ")
                    if user_input == "stop":
                        visualise = 0

                #########
                # Debug #
                #########
                if debug:
                    # Print latest values
                    print("Pulse widths: ", pulse_widths[-1])
                    print("Pulse widths 10%: ", pulse_widths_10[-1])
                    print("Pulse widths 20%: ", pulse_widths_20[-1])
                    print("Pulse widths 30%: ", pulse_widths_30[-1])
                    print("Pulse widths 40%: ", pulse_widths_40[-1])
                    print("Pulse widths 50%: ", pulse_widths_50[-1])
                    print("Pulse widths 60%: ", pulse_widths_60[-1])
                    print("Pulse widths 70%: ", pulse_widths_70[-1])
                    print("Pulse widths 80%: ", pulse_widths_80[-1])
                    print("Pulse widths 90%: ", pulse_widths_90[-1])
                    print("Systolic widths 10%: ", systolic_widths_10[-1])
                    print("Systolic widths 20%: ", systolic_widths_20[-1])
                    print("Systolic widths 30%: ", systolic_widths_30[-1])
                    print("Systolic widths 40%: ", systolic_widths_40[-1])
                    print("Systolic widths 50%: ", systolic_widths_50[-1])
                    print("Systolic widths 60%: ", systolic_widths_60[-1])
                    print("Systolic widths 70%: ", systolic_widths_70[-1])
                    print("Systolic widths 80%: ", systolic_widths_80[-1])
                    print("Systolic widths 90%: ", systolic_widths_90[-1])
                    print("Diastolic widths 10%: ", diastolic_widths_10[-1])
                    print("Diastolic widths 20%: ", diastolic_widths_20[-1])
                    print("Diastolic widths 30%: ", diastolic_widths_30[-1])
                    print("Diastolic widths 40%: ", diastolic_widths_40[-1])
                    print("Diastolic widths 50%: ", diastolic_widths_50[-1])
                    print("Diastolic widths 60%: ", diastolic_widths_60[-1])
                    print("Diastolic widths 70%: ", diastolic_widths_70[-1])
                    print("Diastolic widths 80%: ", diastolic_widths_80[-1])
                    print("Diastolic widths 90%: ", diastolic_widths_90[-1])
                    print("DS ratios 10%: ", ds_ratios_10[-1])
                    print("DS ratios 20%: ", ds_ratios_20[-1])
                    print("DS ratios 30%: ", ds_ratios_30[-1])
                    print("DS ratios 40%: ", ds_ratios_40[-1])
                    print("DS ratios 50%: ", ds_ratios_50[-1])
                    print("DS ratios 60%: ", ds_ratios_60[-1])
                    print("DS ratios 70%: ", ds_ratios_70[-1])
                    print("DS ratios 80%: ", ds_ratios_80[-1])
                    print("DS ratios 90%: ", ds_ratios_90[-1])
                    
        # Add the extracted features to the dictionary
        width_features = {}
        width_features["pulse_width"] = np.nanmedian(pulse_widths)
        width_features["pulse_width_10"] = np.nanmedian(pulse_widths_10)
        width_features["pulse_width_20"] = np.nanmedian(pulse_widths_20)
        width_features["pulse_width_30"] = np.nanmedian(pulse_widths_30)
        width_features["pulse_width_40"] = np.nanmedian(pulse_widths_40)
        width_features["pulse_width_50"] = np.nanmedian(pulse_widths_50)
        width_features["pulse_width_60"] = np.nanmedian(pulse_widths_60)
        width_features["pulse_width_70"] = np.nanmedian(pulse_widths_70)
        width_features["pulse_width_80"] = np.nanmedian(pulse_widths_80)
        width_features["pulse_width_90"] = np.nanmedian(pulse_widths_90)
        width_features["systolic_width_10"] = np.nanmedian(systolic_widths_10)
        width_features["systolic_width_20"] = np.nanmedian(systolic_widths_20)
        width_features["systolic_width_30"] = np.nanmedian(systolic_widths_30)
        width_features["systolic_width_40"] = np.nanmedian(systolic_widths_40)
        width_features["systolic_width_50"] = np.nanmedian(systolic_widths_50)
        width_features["systolic_width_60"] = np.nanmedian(systolic_widths_60)
        width_features["systolic_width_70"] = np.nanmedian(systolic_widths_70)
        width_features["systolic_width_80"] = np.nanmedian(systolic_widths_80)
        width_features["systolic_width_90"] = np.nanmedian(systolic_widths_90)
        width_features["diastolic_width_10"] = np.nanmedian(diastolic_widths_10)
        width_features["diastolic_width_20"] = np.nanmedian(diastolic_widths_20)
        width_features["diastolic_width_30"] = np.nanmedian(diastolic_widths_30)
        width_features["diastolic_width_40"] = np.nanmedian(diastolic_widths_40)
        width_features["diastolic_width_50"] = np.nanmedian(diastolic_widths_50)
        width_features["diastolic_width_60"] = np.nanmedian(diastolic_widths_60)
        width_features["diastolic_width_70"] = np.nanmedian(diastolic_widths_70)
        width_features["diastolic_width_80"] = np.nanmedian(diastolic_widths_80)
        width_features["diastolic_width_90"] = np.nanmedian(diastolic_widths_90)
        width_features["ds_ratio_10"] = np.nanmedian(ds_ratios_10)
        width_features["ds_ratio_20"] = np.nanmedian(ds_ratios_20)
        width_features["ds_ratio_30"] = np.nanmedian(ds_ratios_30)
        width_features["ds_ratio_40"] = np.nanmedian(ds_ratios_40)
        width_features["ds_ratio_50"] = np.nanmedian(ds_ratios_50)
        width_features["ds_ratio_60"] = np.nanmedian(ds_ratios_60)
        width_features["ds_ratio_70"] = np.nanmedian(ds_ratios_70)
        width_features["ds_ratio_80"] = np.nanmedian(ds_ratios_80)
        width_features["ds_ratio_90"] = np.nanmedian(ds_ratios_90)

        #########
        # Debug #
        #########
        if debug:
            # Iterate over the values of the dictionary, print the key and value
            for key, value in width_features.items():
                print(key, value)

        return width_features
    else:
        # Add the extracted features to the dictionary
        width_features = {}
        width_features["pulse_width"] = np.NaN
        width_features["pulse_width_10"] = np.NaN
        width_features["pulse_width_20"] = np.NaN
        width_features["pulse_width_30"] = np.NaN
        width_features["pulse_width_40"] = np.NaN
        width_features["pulse_width_50"] = np.NaN
        width_features["pulse_width_60"] = np.NaN
        width_features["pulse_width_70"] = np.NaN
        width_features["pulse_width_80"] = np.NaN
        width_features["pulse_width_90"] = np.NaN
        width_features["systolic_width_10"] = np.NaN
        width_features["systolic_width_20"] = np.NaN
        width_features["systolic_width_30"] = np.NaN
        width_features["systolic_width_40"] = np.NaN
        width_features["systolic_width_50"] = np.NaN
        width_features["systolic_width_60"] = np.NaN
        width_features["systolic_width_70"] = np.NaN
        width_features["systolic_width_80"] = np.NaN
        width_features["systolic_width_90"] = np.NaN
        width_features["diastolic_width_10"] = np.NaN
        width_features["diastolic_width_20"] = np.NaN
        width_features["diastolic_width_30"] = np.NaN
        width_features["diastolic_width_40"] = np.NaN
        width_features["diastolic_width_50"] = np.NaN
        width_features["diastolic_width_60"] = np.NaN
        width_features["diastolic_width_70"] = np.NaN
        width_features["diastolic_width_80"] = np.NaN
        width_features["diastolic_width_90"] = np.NaN
        width_features["ds_ratio_10"] = np.NaN
        width_features["ds_ratio_20"] = np.NaN
        width_features["ds_ratio_30"] = np.NaN
        width_features["ds_ratio_40"] = np.NaN
        width_features["ds_ratio_50"] = np.NaN
        width_features["ds_ratio_60"] = np.NaN
        width_features["ds_ratio_70"] = np.NaN
        width_features["ds_ratio_80"] = np.NaN
        width_features["ds_ratio_90"] = np.NaN
        return width_features