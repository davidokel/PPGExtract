import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp
import data_methods
from scipy.stats import linregress
from scipy.integrate import trapz

def get_half_widths(data, fs, visualise=0, debug=0):
    data = data.dropna().to_numpy()        
    # Normalise the distal_data and proximal_data using the normalise_data
    data = data_methods.normalise_data(data, 100)
    
    # Calling the get_peaks function from data_methods.py to find the peaks in the data
    peak_points, peaks, troughs = data_methods.get_onsets_v2(data,100,debug=False)

    # Forcing positive values by adding 1
    data = 1 + data

    # Plotting used for debugging
    if debug == 1:
        plt.title("Widths and Prominences")
        plt.plot(data)
        plt.plot(peaks, data[peaks], "x")
        plt.show()

    # Initialising lists for the prominences and half widths
    half_widths = []

    # Checking if there are any peaks in the data
    if len(peaks) != 0:
        # Calculating the half and full widths of the peaks
        half_widths_res = sp.peak_widths(data, peaks, rel_height=0.5) # Calculating the half widths of the peaks using the scipy function peak_widths
        half_widths = sp.peak_widths(data, peaks, rel_height=0.5)[0].tolist()
        
        # Prints used for debugging
        if debug == 1:
            print("Half widths: ", half_widths)

        # Visualising the extracted features
        if visualise == 1:
            fig = plt.figure()
            plt.title("Half peak widths")
            plt.plot(data)
            plt.hlines(*half_widths_res[1:], color="C2", label="half peak widths")
            plt.legend(loc="upper left")

            manager = plt.get_current_fig_manager()
            manager.window.showMaximized()
            #plt.axis('off')
            plt.show()

            print("Half widths:", np.nanmedian(half_widths))
    
        return float(np.nanmedian(half_widths))
    else:
        return np.NaN