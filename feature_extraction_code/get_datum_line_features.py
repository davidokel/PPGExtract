import numpy as np
import matplotlib.pyplot as plt

def get_datum_line_features(window_pulse_data, visualise=0):
    # Calculate the datum line
    # Orientate the data and the datum line so that it is horizontal
    # Calculate the area between the datum line and the data
    # Calculate the median of the data that falls within the bounds of the datum line
    # Calculate the theta needed to rotate the data to make it horizontal
    # If the data is below the datum line, find its argmin and then find the abs difference between the datum line and the data at that point
    # If the data is above the datum line, find its argmax and then find the abs difference between the datum line and the data at that point

    def calculate_line(x1, y1, x2, y2):
        # Generate an array of x-values corresponding to the indices of the data array
        x = np.arange(x1, x2 + 1)

        # Calculate the straight line between the specified coordinates
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        y = m * x + b

        return x, y
    
    import numpy as np

    def calculate_straight_line(x1, y1, x2, y2, x_data, y_data):
        # Calculate the slope of the original line
        m = (y2 - y1) / (x2 - x1)

        angle_rad = np.arctan(m)

        # Calculate the angle between the original line and the horizontal axis
        angle_rad = np.arctan(m)
        angle_deg = np.degrees(angle_rad)

        # Apply the rotation transformation to the line's coordinates
        x_rot = x1 * np.cos(angle_rad) - y1 * np.sin(angle_rad)
        y_rot = x1 * np.sin(angle_rad) + y1 * np.cos(angle_rad)
        x_rot2 = x2 * np.cos(angle_rad) - y2 * np.sin(angle_rad)
        y_rot2 = x2 * np.sin(angle_rad) + y2 * np.cos(angle_rad)

        # Generate an array of x-values corresponding to the indices of the data array
        x = np.arange(x_rot, x_rot2 + 1)

        x_rot_data = x_data * np.cos(angle_rad) - y_data * np.sin(angle_rad)
        y_rot_data = x_data * np.sin(angle_rad) + y_data * np.cos(angle_rad)

        # Calculate the rotated line
        y = np.zeros_like(x)  # Initialize an array for y-values
        y[:] = y_rot  # Set all y-values to the rotated y1 value

        return x, y, x_rot_data, y_rot_data

    data = window_pulse_data["raw_pulse_data"]
    pulse_start = 0
    pulse_end = len(data)-1
    peak = window_pulse_data["Relative_peak"]

    # Calculate the straight line between the peak index and the end of the pulse
    peak_coordinates = (peak, data[peak])
    pulse_end_coordinates = (pulse_end, data[pulse_end])
    pulse_start_coordinates = (pulse_start, data[pulse_start])
    
    # Calculate the straight line between the peak index and the end of the pulse
    x_peak, y_peak = peak_coordinates
    x_start, y_start = pulse_start_coordinates
    x_end, y_end = pulse_end_coordinates

    # Calculate the straight line between the peak index and the end of the pulse
    x_datum_start, y_datum_start = calculate_line(x_start, y_start, x_peak, y_peak)
    # Calculate the straight line between the peak index and the end of the pulse
    x_peak_end, y_peak_end= calculate_line(x_peak, y_peak, x_end, y_end)

    # Isolate the data that falls between the datum lines
    datum_start_data = data[x_datum_start[0]:x_datum_start[-1]]
    datum_end_data = data[x_peak_end[0]:x_peak_end[-1]]

    # Isolate the x and y values for the datum_start_data and datum_end_data
    x_datum_start_data = x_datum_start[0:len(datum_start_data)]
    y_datum_start_data = datum_start_data

    x_peak_end_data = x_peak_end[0:len(datum_end_data)]
    y_peak_end_data = datum_end_data

    x_datum_start_rotated, y_datum_start_rotated, x_datum_start_data_rotated, y_datum_start_data_rotated = calculate_straight_line(x_start, y_start, x_peak, y_peak, x_datum_start_data, y_datum_start_data)
    x_datum_end_rotated, y_datum_end_rotated, x_peak_end_data_rotated, y_peak_end_data_rotated = calculate_straight_line(x_peak, y_peak, x_end, y_end, x_peak_end_data, y_peak_end_data)

    # Plot the rotated datum line and the rotated data
    plt.plot(x_datum_start_data_rotated, y_datum_start_data_rotated, 'g--', label = "SP Datum Line Rotated")
    plt.plot(x_peak_end_data_rotated, y_peak_end_data_rotated, 'r--', label = "EP Datum Line Rotated")
    plt.show()

    plt.plot(x_datum_start_rotated, y_datum_start_rotated, 'g--', label = "SP Datum Line Rotated")
    plt.plot(x_datum_end_rotated, y_datum_end_rotated, 'r--', label = "EP Datum Line Rotated")
    plt.show()


    # Plot the rotated datum line
    plt.plot(x_datum_start_rotated, y_datum_start_rotated, 'g--', label = "SP Datum Line Rotated")
    plt.show()

    # Plot the straight line between the peak index and the end of the pulse
    plt.plot(data)
    plt.plot(x_datum_start, y_datum_start, 'g--', label = "SP Datum Line")
    plt.plot(x_peak_end, y_peak_end, 'r--', label = "EP Datum Line")
    plt.show()








    


