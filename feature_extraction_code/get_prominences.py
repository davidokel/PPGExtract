import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp
from support_code.data_methods import data_scaler

def get_prominences(window_pulse_data, visualise=False, debug=False):
    """
    Calculates the prominences of pulses within a given window.

    Inputs:
    - window_pulse_data: A dictionary containing pulse data for each pulse within the window.
                         Each key represents a pulse, and the corresponding value is a dictionary
                         containing the pulse's raw data and peak information.
    - visualise: An optional parameter (default is False) to visualize the amplitude plots.
    - debug: An optional parameter (default is False) to print the prominence values.

    Output:
    - Median Prominence: The median of all calculated prominences.

    Error Handling:
    - If the window_pulse_data is empty, the function returns NaN.
    """

    ############################################################
    # Initialising list to store the extracted PROMINENCE data #
    ############################################################
    prominences = []

    # If the window_pulse_data is not empty
    if window_pulse_data:
        # Iterate over the keys of the dictionary, every key represents a pulse within the window
        for key in window_pulse_data:
            #######################
            # Defining pulse data #
            #######################
            data = window_pulse_data[key]["pulse_data"]
            peak = np.array([window_pulse_data[key]["Relative_peak"]])
            
            """Scaling the data to have a minimum value of 0 by adding a scaling factor.
            The scaling process shifts the entire signal vertically, but it does not change the relative amplitudes or the shape of the waveform. 
            The prominence calculation relies on the relative heights of the peaks in the signal, and scaling the data does not alter these relative heights."""
            data = data_scaler(data)

            # Checking if there are any peaks in the data
            if peak:
                ########################################
                # Calculate of prominence of the pulse #
                ########################################
                prominence = sp.peak_prominences(data, peak)[0]
                # Adding the prominences to the list
                prominences.append(prominence)

                contour_heights = data[peak] - prominence # Calculating the contour heights (Used for plotting)

                ############
                # Plotting #
                ############
                if visualise:
                    plt.title("Pulse Prominence")
                    plt.plot(data)
                    plt.plot(peak,data[peak],'x')
                    plt.vlines(x=peak, ymin=contour_heights, ymax=data[peak], label="amplitudes", linestyles="dashed", colors="green")
                    plt.legend(loc="upper left")
                    """manager = plt.get_current_fig_manager()
                    manager.window.showMaximized()"""
                    #plt.axis('off')
                    plt.show()

                    # Get the user input to see if they want to move onto the next visualisation or stop visualising
                    user_input = input("Press enter to continue, or type 'stop' to stop visualising: ")
                    if user_input == "stop":
                        visualise = 0
                
                #########
                # Debug #
                #########
                if debug:
                    print("Prominence: " + str(prominences[-1]))
                    print("\n")

        return float(np.nanmedian(prominences))
    else:
        return np.NaN