import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp
import data_methods

def get_amplitudes_widths_prominences(data,fs,visualise=0, debug = 0):
    peaks = data_methods.get_peaks(data, fs) # Given data as input and frequency of data, find the locations of the peaks

    # Plotting used for debugging
    if debug == 1:
        plt.plot(data)
        plt.plot(peaks, data[peaks], "x")
        plt.show()

    # LISTS FOR STORING RESULTS
    prominences = []
    half_widths = []

    # Simplified quality assessment, if there are less peaks than 1/4 of the data length (in seconds) then don't extract features
    seconds = int((len(data)/fs)*0.25)

    if (len(peaks) > seconds) == True:
        # Calculating the prominences and half_widths of the peaks
        prominences.append((sp.peak_prominences(data, peaks)[0]).tolist())
        prominences = [item for sublist in prominences for item in sublist] # Flattening the list of lists into a list of values
        contour_heights = data[peaks] - prominences # Calculating the contour heights (Used for plotting)

        # Calculating the half and full widths of the peaks
        half_widths_res = sp.peak_widths(data, peaks, rel_height=0.5)
        half_widths = sp.peak_widths(data, peaks, rel_height=0.5)[0].tolist() # Calculating the half peak widths
        
        # Prints used for debugging
        if debug == 1:
            print("Prominences: ", prominences)
            print("Half widths: ", half_widths)

        if visualise == 1:
            fig = plt.figure()
            plt.subplot(2,1,1)
            plt.title("Amplitudes")
            plt.plot(data)
            plt.plot(peaks,data[peaks],'x')
            plt.vlines(x=peaks, ymin=contour_heights, ymax=data[peaks], label="amplitudes", linestyles="dashed", colors="green")
            plt.legend(loc="upper left")
            plt.axis('off')

            plt.subplot(2,1,2)
            plt.title("Half peak widths")
            plt.plot(data)
            plt.hlines(*half_widths_res[1:], color="C2", label="half peak widths")
            plt.legend(loc="upper left")

            manager = plt.get_current_fig_manager()
            manager.window.showMaximized()
            plt.axis('off')
            plt.show()

            print("Prominences:", np.nanmedian(prominences))
            print("Half peak widths:", np.nanmedian(half_widths))
    
        return float(np.nanmedian(prominences)), float(np.nanmedian(half_widths))
    else:
        return np.NaN, np.NaN
