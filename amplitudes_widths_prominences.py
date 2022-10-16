import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp
from data_methods import *

def get_amplitudes_widths_prominences(data,fs,visualise=0):
    peaks = get_peaks(data, fs)

    # LISTS FOR STORING RESULTS
    prominences = []
    half_widths = []

    if len(peaks) != 0:
        # ADDING THE PROMINENCES TO THE LIST prominences
        prominences.append((sp.peak_prominences(data, peaks)[0]).tolist())
        prominences = [item for sublist in prominences for item in sublist] # Flattening the list of lists into a list of values
        contour_heights = data[peaks] - prominences # Calculating the contour heights (Used for plotting)

        # CALCULATING THE HALF PEAK AND FULL PEAK WIDTHS
        half_widths_res = sp.peak_widths(data, peaks, rel_height=0.5)
        half_widths = sp.peak_widths(data, peaks, rel_height=0.5)[0].tolist()

        if visualise == 1:
            fig = plt.figure()
            plt.subplot(2,1,1)
            plt.title("Amplitudes")
            plt.plot(data)
            plt.plot(peaks,data[peaks],'x')
            plt.vlines(x=peaks, ymin=contour_heights, ymax=data[peaks], label="amplitudes", linestyles="dashed", colors="green")
            plt.legend(loc="upper left")

            plt.subplot(2,1,2)
            plt.title("Half peak widths")
            plt.plot(data)
            plt.hlines(*half_widths_res[1:], color="C2", label="half peak widths")
            plt.legend(loc="upper left")

            manager = plt.get_current_fig_manager()
            manager.window.showMaximized()
            plt.show()

        return np.nanmedian(prominences), np.nanmedian(half_widths)/fs
    else:
        return np.NaN, np.NaN
