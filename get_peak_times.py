import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp
import data_methods
from scipy.stats import linregress
from scipy.integrate import trapz

def get_peak_times(data, fs, visualise=0, debug=0):
    data = data.dropna().to_numpy()        
    # Normalise the distal_data and proximal_data using the normalise_data
    data = data_methods.normalise_data(data, 100)

    # Calling the get_peaks function from data_methods.py to find the peaks in the data
    peak_points, peaks, troughs = data_methods.get_onsets_v2(data,100,debug=False)

    rise_times = []
    decay_times = []

    if len(peaks) != 0:
        #peak_points = data_methods.get_onsets(data, peaks) # Given the data and the peak locations, get the onset locations
        
        # Printing and plotting used for debugging
        if debug == 1:
            plt.title("Upslopes, downslopes, rise times, decay times, AUC, AUC ratios")
            plt.plot(data)
            plt.plot(peaks, data[peaks], "x")
            plt.show()

        for key in peak_points:
            peak_loc = peak_points[key]["Peak"]

            pre_loc = peak_points[key]["Pre_peak"]
            rise_time = abs(peak_loc-pre_loc)
            rise_times.append(rise_time)

            post_loc = peak_points[key]["Post_peak"]
            decay_time = abs(post_loc-peak_loc)
            decay_times.append(decay_time)

        if visualise == 1:
            plt.subplot(2,1,1)
            plt.title("Rise times")
            plt.plot(data)
            for key in peak_points:
                peak = peak_points[key]["Peak"]
                pre = peak_points[key]["Pre_peak"]

                plt.annotate(text = "", xy=(peak,data[pre]), xytext=(peak,data[peak]), arrowprops=dict(arrowstyle='-'))
                plt.annotate(text = "", xy=(pre,data[pre]), xytext=(peak,data[pre]), arrowprops=dict(arrowstyle='<->'))
            #plt.axis('off')
            plt.subplot(2,1,2)
            plt.title("Decay times")
            plt.plot(data)
            for key in peak_points:
                peak = peak_points[key]["Peak"]
                post = peak_points[key]["Post_peak"]

                plt.annotate(text = "", xy=(peak,data[post]), xytext=(peak,data[peak]), arrowprops=dict(arrowstyle='-'))
                plt.annotate(text = "", xy=(post,data[post]), xytext=(peak,data[post]), arrowprops=dict(arrowstyle='<->'))
            
            manager = plt.get_current_fig_manager()
            manager.window.showMaximized()
            #plt.axis('off')
            plt.show()

        return float(np.nanmedian(rise_times)), float(np.nanmedian(decay_times))
    else:
        return np.NaN, np.NaN