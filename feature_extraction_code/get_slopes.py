import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import math
import matplotlib.patches as patches

def get_slopes(window_pulse_data, visualise=False, debug=False):
    """
    Calculates various slope-related features from pulse waveform data.

    Inputs:
    - window_pulse_data: A dictionary containing pulse data for each pulse within the window.
                         Each key represents a pulse, and the corresponding value is a dictionary
                         containing the pulse's raw data and peak information.
    - visualise: An optional parameter (default is False) to visualize the upslopes and downslopes.
    - debug: An optional parameter (default is False) to print debug information.

    Outputs:
    - features: A dictionary containing the calculated slope-related features.

    Note: If the window_pulse_data is empty, the function returns None.

    """
    ###########################################################
    # Initialising a dictionary and lists to store SLOPE data #
    ###########################################################
    slope_features = {}
    upslope_lengths, downslope_lengths, upslopes, downslopes, onset_end_slopes, upslope_downslope_ratios, pulse_length_height_ratios, upslope_downslope_length_ratios, upslope_pulse_length_ratios, downslope_pulse_length_ratios = [], [], [], [], [], [], [], [], [], []

    # If the window_pulse_data is not empty
    if window_pulse_data:
        # Iterate over the keys of the dictionary, every key represents a pulse within the window
        for key in window_pulse_data:
            #######################
            # Defining pulse data #
            #######################
            data = window_pulse_data[key]["pulse_data"]
            peak = window_pulse_data[key]["Relative_peak"]
            pre = 0
            post = len(data) -1
            pulse_length = post - pre

            if peak:
                ############################
                # UPSLOPE DOWNSLOPE LENGTH #
                ############################
                # Calculating the distance between two points in order to find the straight line for the upslope and downslope
                # distance = sqrt((x2 - x1)^2 + (y2 - y1)^2)
                upslope_length = np.sqrt((peak - pre)**2 + (data[peak] - data[pre])**2)
                downslope_length = np.sqrt((post - peak)**2 + (data[-1] - data[peak])**2)
                upslope_lengths.append(upslope_length)
                downslope_lengths.append(downslope_length)
                
                ###########
                # UPSLOPE #
                ###########
                # The upslope measures the rate of change of the pulse during the rising phase, from the onset to the peak. 
                # This can give an indication of how quickly the pulse is rising and how steep the rising edge is.
                # Calculate the angle of the line between the onset and the peak against the horizontal axis
                # The angle is in radians, so convert to degrees
                upslope, _, _, _, _ = linregress([pre,peak],[data[pre],data[peak]])
                upslope = (upslope * 100)
                upslopes.append(upslope)

                #############
                # DOWNSLOPE #
                #############
                # The downslope measures the rate of change of the pulse during the falling phase, from the peak to the end of the pulse. 
                # This can give an indication of how quickly the pulse is falling and how steep the falling edge is.
                downslope, _, _, _, _ = linregress([peak,post],[data[peak],data[-1]])
                downslope = (downslope * 100)
                downslopes.append(downslope)

                ###################
                # ONSET_END_SLOPE #
                ###################
                # The onset-end slope measures the rate of change of the pulse over the entire pulse length, from the onset to the end of the pulse. 
                # This can give an indication of the overall shape of the pulse and how steep it is.
                onset_end_slope, _, _, _, _  = linregress([pre, post], [data[pre], data[-1]])
                onset_end_slopes.append(onset_end_slope)

                ###########################
                # UPSLOPE_DOWNSLOPE_RATIO #
                ###########################
                # This ratio can give an indication of how steep the pulse is on the rising edge (upslope) compared to the falling edge (downslope). 
                # If the ratio is greater than 1, then the pulse rises more steeply than it falls, and if the ratio is less than 1, then the pulse falls more steeply than it rises. 
                # This could be useful for comparing different pulses to see which ones have a more pronounced rising or falling edge.
                upslope_downslope_ratio = upslope / downslope if downslope != 0 else np.nan
                upslope_downslope_ratios.append(upslope_downslope_ratio)

                #############################
                # PULSE_LENGTH_HEIGHT_RATIO #
                #############################
                # This ratio can give an indication of the overall shape of the pulse, specifically how long it takes to return to baseline after reaching its peak.
                # If the ratio is high, then the pulse takes a relatively long time to return to baseline after reaching its peak, indicating a broader shape. 
                # If the ratio is low, then the pulse returns quickly to baseline, indicating a narrower shape. 
                # This could be useful for comparing different pulses to see if there are consistent differences in the shape of the pulse.
                pulse_length_height_ratio = pulse_length / (data[peak] - data[pre]) if data[peak] != data[pre] else np.nan
                pulse_length_height_ratios.append(pulse_length_height_ratio)

                ##################################
                # UPSLOPE_DONWSLOPE_LENGTH_RATIO #
                ##################################
                # This ratio can help to quantify the relative contribution of the rising phase of the pulse to the overall length of the pulse. 
                # A pulse with a higher ratio would have a longer rising phase relative to its overall length, while a pulse with a lower ratio would have a shorter rising phase. 
                upslope_downslope_length_ratio = upslope_length / downslope_length if downslope_length != 0 else np.nan
                upslope_downslope_length_ratios.append(upslope_downslope_length_ratio)
                
                ##############################
                # UPSLOPE_PULSE_LENGTH_RATIO #
                ##############################
                # This ratio measures the proportion of the total pulse length that is made up of the rising phase. 
                # This can be useful in characterizing the slope of the rising phase and how it contributes to the overall shape of the pulse. 
                # For example, a pulse with a higher ratio may have a steeper or more pronounced rise, while a pulse with a lower ratio may have a more gradual rise.
                upslope_pulse_length_ratio = upslope_length / pulse_length if pulse_length != 0 else np.nan
                upslope_pulse_length_ratios.append(upslope_pulse_length_ratio)

                ################################
                # PULSE_DOWNSLOPE_LENGTH_RATIO #
                ################################
                # This ratio measures the proportion of the total pulse length that is made up of the falling phase, from the peak to the end of the pulse. 
                # This can be useful in characterizing the slope of the falling phase and how it contributes to the overall shape of the pulse. 
                # For example, a pulse with a higher ratio may have a steeper or more pronounced fall, while a pulse with a lower ratio may have a more gradual fall.
                downslope_pulse_length_ratio = downslope_length / pulse_length if pulse_length != 0 else np.nan
                downslope_pulse_length_ratios.append(downslope_pulse_length_ratio)

                ############
                # Plotting #
                ############
                if visualise:
                    plt.subplot(2,1,1)
                    plt.title("Upslope & Downslope Lengths (samples)")
                    plt.plot(data)
                    # Annotate the plot with the upslope length, add the length to the arrow as text
                    plt.annotate(text = "", xy=(pre,data[pre]), xytext=(peak,data[peak]), arrowprops=dict(arrowstyle= '<|-|>', color='green', lw=3, ls='--'), label="Upslope")
                    plt.annotate(text = "", xy=(peak,data[peak]), xytext=(post,data[-1]), arrowprops=dict(arrowstyle= '<|-|>', color='red', lw=3, ls='--'), label="Downslope")
                    # Get the x, y coordinates for the midpoint between the pre and peak, and the peak and post
                    x1 = (pre + peak) / 2
                    y1 = (data[pre] + data[peak]) / 2

                    x2 = (peak + post) / 2
                    y2 = (data[peak] + data[-1]) / 2

                    # Add text within a box with a grey background to the midpoint between the pre and peak, and the peak and post with te upslope and downslope lengths
                    plt.text(x1, y1, "Upslope Length: " + str(upslope_lengths[-1]), bbox=dict(facecolor='grey', alpha=0.5))
                    plt.text(x2, y2, "Downslope Length: " + str(downslope_lengths[-1]), bbox=dict(facecolor='grey', alpha=0.5))

                    
                    plt.subplot(2,1,2)
                    plt.title("Upslope & Downslope (%)")
                    plt.plot(data)
                    # Annotate the plot with the angles of the upslope and downslope
                    plt.annotate(text = "", xy=(pre,data[pre]), xytext=(peak,data[peak]), arrowprops=dict(arrowstyle= '<|-|>', color='green', lw=3, ls='--'), label="Upslope")
                    plt.annotate(text = "", xy=(peak,data[peak]), xytext=(post,data[-1]), arrowprops=dict(arrowstyle= '<|-|>', color='red', lw=3, ls='--'), label="Downslope")
                    
                    # Add text within a box with a grey background to the midpoint between the pre and peak, and the peak and post with te upslope and downslope angles
                    plt.text(x1, y1, "Upslope (%): " + str(upslopes[-1]), bbox=dict(facecolor='grey', alpha=0.5))
                    plt.text(x2, y2, "Downslope (%): " + str(downslopes[-1]), bbox=dict(facecolor='grey', alpha=0.5))

                    plt.show()

                    # Get the user input to see if they want to move onto the next visualisation or stop visualising
                    user_input = input("Press enter to continue, or type 'stop' to stop visualising: ")
                    if user_input == "stop":
                        visualise = 0

                #########
                # Debug #
                #########
                if debug: 
                    print("Upslope Length: ", upslope_lengths[-1])
                    print("Downslope Length: ", downslope_lengths[-1])
                    print("Upslope: ", upslopes[-1])
                    print("Downslope: ", downslopes[-1])
                    print("Onset-End Slope: ", onset_end_slopes[-1])
                    print("Upslope-Downslope Ratio: ", upslope_downslope_ratios[-1])
                    print("Pulse Length-Height Ratio: ", pulse_length_height_ratios[-1])
                    print("Upslope-Downslope Length Ratio: ", upslope_downslope_length_ratios[-1])
                    print("Upslope-Pulse Length Ratio: ", upslope_pulse_length_ratios[-1])
                    print("Downslope-Pulse Length Ratio: ", downslope_pulse_length_ratios[-1])
                    print("\n")

        # Calculating the median of the extracted features
        slope_features["median_upslope_length"] = np.nanmedian(upslope_lengths)
        slope_features["median_downslope_length"] = np.nanmedian(downslope_lengths)
        slope_features["median_upslope"] = np.nanmedian(upslopes)
        slope_features["median_downslope"] = np.nanmedian(downslopes)
        slope_features["median_onset_end_slope"] = np.nanmedian(onset_end_slopes)
        slope_features["median_upslope_downslope_ratio"] = np.nanmedian(upslope_downslope_ratios)
        slope_features["median_pulse_length_height_ratio"] = np.nanmedian(pulse_length_height_ratios)
        slope_features["median_upslope_downslope_length_ratio"] = np.nanmedian(upslope_downslope_length_ratios)
        slope_features["median_upslope_pulse_length_ratio"] = np.nanmedian(upslope_pulse_length_ratios)
        slope_features["median_downslope_pulse_length_ratio"] = np.nanmedian(downslope_pulse_length_ratios)
        return slope_features 
    else:
        return np.NaN