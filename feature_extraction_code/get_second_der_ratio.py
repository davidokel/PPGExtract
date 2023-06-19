import numpy as np

def get_second_der_ratio(window_pulse_data, debug=False):
    """
    Calculates the median of the ratio between the maximum and minimum values of the second derivative of pulse waveform data.

    Inputs:
    - window_pulse_data: A dictionary containing pulse data for each pulse within the window.
                         Each key represents a pulse, and the corresponding value is a dictionary
                         containing the pulse's raw data.
    - debug: An optional parameter (default is False) to print debug information.

    Outputs:
    - median_ratio: The median of the calculated second derivative ratios.

    Note: If the window_pulse_data is empty, the function returns NaN.

    """

    #########################################################
    # Initialising a list to store second derivative ratios #
    #########################################################
    second_derivative_ratios = []

    # If the window_pulse_data is not empty
    if window_pulse_data:
        # Iterate over the keys of the dictionary, every key represents a pulse within the window
        for key in window_pulse_data:
            #######################
            # Defining pulse data #
            #######################
            data = window_pulse_data[key]["pulse_data"]

            ####################################
            # Calculating second derivatives #
            ####################################
            second_derivative = np.gradient(np.gradient(data))

            ###################################
            # Calculating maximum and minimum #
            ###################################
            # Finding the maximum and minimum values of the second derivative.
            max_value = max(second_derivative)
            min_value = min(second_derivative)

            ########################################################
            # Calculating the ratio of maximum and minimum values #
            ########################################################
            # Taking the absolute value of the ratio between the maximum and minimum values of the second derivative.
            second_derivative_ratio = abs(max_value / min_value)

            #####################################################
            # Storing the calculated ratio in the list of ratios #
            #####################################################
            second_derivative_ratios.append(second_derivative_ratio)

        ###############################################################
        # Calculating the median of the second derivative ratios list #
        ###############################################################
        # The median provides a representative value for the distribution of ratios.
        median_ratio = np.nanmedian(second_derivative_ratios)
        
        #########
        # Debug #
        #########
        if debug:
            print("Second Derivative Ratios: ", second_derivative_ratios)
            print("Median Ratio: ", median_ratio)

        return median_ratio
    else:
        return np.NaN
