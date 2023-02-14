import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp
import data_methods

def get_prominences(data, fs, visualise=0, debug=0):
    # Calling the get_peaks function from data_methods.py to find the peaks in the data
    peaks = data_methods.get_peaks(data, fs)

    # Forcing positive values by adding 1
    data = 1 + data

    # Plotting used for debugging
    if debug == 1:
        plt.title("Prominences")
        plt.plot(data)
        plt.plot(peaks, data[peaks], "x")
        plt.show()

    # Initialising lists for the prominences and half widths
    prominences = []

    # Checking if there are any peaks in the data
    if len(peaks) != 0:

        # Calculating the prominences and half_widths of the peaks
        prominences.append((sp.peak_prominences(data, peaks)[0]).tolist()) # Calculating the prominences of the peaks using the scipy function peak_prominences
        prominences = [item for sublist in prominences for item in sublist] # Flattening the list of lists into a list of values
        contour_heights = data[peaks] - prominences # Calculating the contour heights (Used for plotting)
        
        # Prints used for debugging
        if debug == 1:
            print("Prominences: ", prominences)

        # Visualising the extracted features
        if visualise == 1:
            fig = plt.figure()
            plt.title("Amplitudes")
            plt.plot(data)
            plt.plot(peaks,data[peaks],'x')
            plt.vlines(x=peaks, ymin=contour_heights, ymax=data[peaks], label="amplitudes", linestyles="dashed", colors="green")
            plt.legend(loc="upper left")
            manager = plt.get_current_fig_manager()
            manager.window.showMaximized()
            #plt.axis('off')
            plt.show()

            print("Prominences:", np.nanmedian(prominences))
    
        return float(np.nanmedian(prominences))
    else:
        return np.NaN