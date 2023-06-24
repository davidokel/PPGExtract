import matplotlib.pyplot as plt
import numpy as np

def get_peak_times(window_pulse_data, visualise=0):
    """
    Function to extract rise and decay times from pulse data.

    Parameters:
    - window_pulse_data: dictionary containing pulse data for each window
    - visualise: flag to indicate whether to visualize the pulse data

    Returns:
    - Median rise time.
    - Median decay time.
    - Median rise/decay time ratio.

    Error Handling:
    - If the window_pulse_data is empty, the function returns NaN for all outputs.
    """

    ##################################################################
    # Initialising lists to store the extracted RISE/DECAY time data #
    ##################################################################
    rise_times, decay_times, rise_decay_time_ratios = [], [], []

    # Check that window_pulse_data is not empty
    if window_pulse_data:
        # Iterating over the keys of the dictionary, every key represents a pulse within the window
        for key in window_pulse_data:
            #######################
            # Defining pulse data #
            #######################
            pulse_data = window_pulse_data[key]
            data = pulse_data["pulse_data"]
            peak = pulse_data["relative_peak"]
            pre = 0
            post = len(data)
            
            ######################################
            # Calculate rise time and decay time #
            ######################################
            if peak:
                # Calculate rise time as the difference between the peak and the pre-peak
                rise_time = abs(peak - pre)
                rise_times.append(rise_time)

                # Calculate decay time as the difference between the peak and the post-peak
                decay_time = abs(post - peak)
                decay_times.append(decay_time)

                # Calculate rise/decay time ratio
                rise_decay_time_ratio = rise_time / decay_time
                rise_decay_time_ratios.append(rise_decay_time_ratio)
                
                ############
                # Plotting #
                ############
                if visualise == 1:
                    plt.subplot(2,1,1)
                    plt.title("Rise times")
                    plt.plot(data)
                    plt.annotate(text="", xy=(peak,data[pre]), xytext=(peak,data[peak]), arrowprops=dict(arrowstyle='-'))
                    plt.annotate(text="", xy=(pre,data[pre]), xytext=(peak,data[pre]), arrowprops=dict(arrowstyle='<->'))
                    
                    plt.subplot(2,1,2)
                    plt.title("Decay times")
                    plt.plot(data)
                    plt.annotate(text="", xy=(peak,data[-1]), xytext=(peak,data[peak]), arrowprops=dict(arrowstyle='-'))
                    plt.annotate(text="", xy=(post,data[-1]), xytext=(peak,data[-1]), arrowprops=dict(arrowstyle='<->'))
                    
                    manager = plt.get_current_fig_manager()
                    manager.window.showMaximized()
                    plt.show()

                    # Get the user input to see if they want to move onto the next visualisation or stop visualising
                    user_input = input("Press enter to continue, or type 'stop' to stop visualising: ")
                    if user_input == "stop":
                        visualise = 0

        time_features = {}
        time_features["rise_times"] = float(np.nanmedian(rise_times))
        time_features["decay_times"] = float(np.nanmedian(decay_times))
        time_features["rise_decay_time_ratios"] = float(np.nanmedian(rise_decay_time_ratios))
        return time_features
    # Return NaN if no peaks are found
    else:
        time_features = {}
        time_features["rise_times"] = np.NaN
        time_features["decay_times"] = np.NaN
        time_features["rise_decay_time_ratios"] = np.NaN
        return time_features