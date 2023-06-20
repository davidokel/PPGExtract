import numpy as np
import matplotlib.pyplot as plt

def get_datum_line_features(window_pulse_data, visualise=False, debug=False):
    """
    Calculates various features related to the datum lines in pulse waveform data.

    Inputs:
    - window_pulse_data: A dictionary containing pulse data for each pulse within the window.
                         Each key represents a pulse, and the corresponding value is a dictionary
                         containing the pulse's raw data.
    - visualise: An optional parameter (default is False) to visualize the datum lines and related data plots.
    - debug: An optional parameter (default is False) to print debug information.

    Outputs:
    - datum_features: A dictionary containing calculated features related to the datum lines.

    Note: If the window_pulse_data is empty, the function returns NaN.

    """
        
    ##############################################################
    # Function to calculate the straight line between two points #
    ##############################################################
    def calculate_line(x1, y1, x2, y2):
        # Generate an array of x-values corresponding to the indices of the data array
        x = np.arange(x1, x2 + 1)

        # Calculate the straight line between the specified coordinates
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        y = m * x + b

        return x, y
    
    ################################################
    # Initialising a list to store datum line data #
    ################################################
    datum_features = {}
    start_datum_areas, end_datum_areas, datum_area_ratios, max_end_datum_differences, max_start_datum_differences, median_start_datum_differences, median_end_datum_differences = [], [], [], [], [], [], []

    # If the window_pulse_data is not empty
    if window_pulse_data:
        # Iterate over the keys of the dictionary, every key represents a pulse within the window
        for key in window_pulse_data:
            #######################
            # Defining pulse data #
            #######################
            data = window_pulse_data[key]["pulse_data"]
            peak = window_pulse_data[key]["relative_peak"]
            pulse_start = 0
            pulse_end = len(data)-1
            
            # Getting the coordinates of the peak, start and end of the pulse
            peak_coordinates = (peak, data[peak])
            pulse_end_coordinates = (pulse_end, data[pulse_end])
            pulse_start_coordinates = (pulse_start, data[pulse_start])
            
            # Getting the x and y coordinates of the peak, start and end of the pulse
            x_peak, y_peak = peak_coordinates
            x_start, y_start = pulse_start_coordinates
            x_end, y_end = pulse_end_coordinates

            ###############################
            # Calculating the datum lines #
            ###############################
            x_datum_line_start, y_datum_line_start = calculate_line(x_start, y_start, x_peak, y_peak)
            x_datum_line_end, y_datum_line_end = calculate_line(x_peak, y_peak, x_end, y_end)

            # Isolating the data between the datum lines
            start_datum_data = data[pulse_start:peak]
            start_datum_data = np.append(start_datum_data, data[peak])
            end_datum_data = data[peak:pulse_end]
            end_datum_data = np.append(end_datum_data, data[pulse_end])

            ###############################################
            # Calculating the features of the datum lines #
            ###############################################
            # Calculate the area between the datum lines and the respective data
            start_datum_area = np.trapz(start_datum_data, x_datum_line_start)
            end_datum_area = np.trapz(end_datum_data, x_datum_line_end)
            start_datum_areas.append(start_datum_area)
            end_datum_areas.append(end_datum_area)
            
            # Calculate the median difference between the datum lines and their respective data
            start_datum_difference = np.subtract(start_datum_data, y_datum_line_start)
            end_datum_difference = np.subtract(end_datum_data, y_datum_line_end)

            median_start_datum_difference = np.median(start_datum_difference)
            median_end_datum_difference = np.median(end_datum_difference)
            median_start_datum_differences.append(median_start_datum_difference)
            median_end_datum_differences.append(median_end_datum_difference)
            
            # Finding the maximum difference between the datum lines and their respective data
            max_start_datum_difference = np.max(start_datum_difference)
            max_end_datum_difference = np.max(end_datum_difference)
            max_start_datum_differences.append(max_start_datum_difference)
            max_end_datum_differences.append(max_end_datum_difference)

            # Ratio between the start_datum_area and end_datum_area
            datum_area_ratio = start_datum_area / end_datum_area
            datum_area_ratios.append(datum_area_ratio)
            
            ############
            # Plotting #
            ############
            if visualise:
                # Create the figure and subplots
                fig = plt.figure(figsize=(10, 8))

                # Create gridspec to define the layout of subplots
                gs = fig.add_gridspec(2, 2)

                # Define the positions of each subplot
                ax1 = fig.add_subplot(gs[:, 0])
                ax2 = fig.add_subplot(gs[0, 1])
                ax3 = fig.add_subplot(gs[1, 1])

                # Plot something in each subplot
                ax1.plot(data, label="Raw data")
                ax1.plot(x_datum_line_start, y_datum_line_start, "g--", label="Datum line start")
                ax1.plot(x_datum_line_end, y_datum_line_end, "r--", label="Datum line end")
                ax1.set_title('Datum lines')

                ax2.plot(x_datum_line_start, y_datum_line_start, "g--", label="Datum line start")
                ax2.fill_between(x_datum_line_start, y_datum_line_start, start_datum_data, alpha=0.5, color='green')
                ax2.set_title('Area between the Start datum line and the respective data')

                ax3.plot(x_datum_line_end, y_datum_line_end, "r--", label="Datum line end")
                ax3.fill_between(x_datum_line_end, y_datum_line_end, end_datum_data, alpha=0.5, color='red')
                ax3.set_title('Area between the End datum line and the respective data')
                plt.show()

                # Get the user input to see if they want to move onto the next visualisation or stop visualising
                user_input = input("Press enter to continue, or type 'stop' to stop visualising: ")
                if user_input == "stop":
                    visualise = 0

            if debug:
                print("Start datum area: ", start_datum_areas[-1])
                print("End datum area: ", end_datum_areas[-1])
                print("Datum area ratio: ", datum_area_ratios[-1])
                print("Max start datum difference: ", max_start_datum_differences[-1])
                print("Max end datum difference: ", max_end_datum_differences[-1])
                print("Median start datum difference: ", median_start_datum_differences[-1])
                print("Median end datum difference: ", median_end_datum_differences[-1])
                print("\n")
        
        datum_features["Start datum area"] = np.nanmedian(start_datum_areas)
        datum_features["End datum area"] = np.nanmedian(end_datum_areas)
        datum_features["Datum area ratio"] = np.nanmedian(datum_area_ratios)
        datum_features["Max start datum difference"] = np.nanmedian(max_start_datum_differences)
        datum_features["Max end datum difference"] = np.nanmedian(max_end_datum_differences)
        datum_features["Median start datum difference"] = np.nanmedian(median_start_datum_differences)
        datum_features["Median end datum difference"] = np.nanmedian(median_end_datum_differences)
        return datum_features
    else:
        return np.nan












    


