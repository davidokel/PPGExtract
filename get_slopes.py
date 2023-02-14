import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp
import data_methods
from scipy.stats import linregress
from scipy.integrate import trapz

def get_slopes(data,fs,visualise=0, debug = 0):
    data = data.dropna().to_numpy()        
    # Normalise the distal_data and proximal_data using the normalise_data
    data = data_methods.normalise_data(data, 100)
    
    # Calling the get_peaks function from data_methods.py to find the peaks in the data
    peaks = data_methods.get_peaks(data, fs) # Given the data and the sampling frequency, get the peak locations
    data_scaled = data_methods.data_scaler(data) # Scale the data to be between 0 and 1 (Used for plotting)

    upslopes = []
    downslopes = []

    if len(peaks) != 0:
        peak_points = data_methods.get_onsets(data, peaks) # Given the data and the peak locations, get the onset locations
        
        # Printing and plotting used for debugging
        if debug == 1:
            plt.title("Upslopes and downslopes")
            plt.plot(data)
            plt.plot(peaks, data[peaks], "x")
            plt.show()

        for key in peak_points:
            peak_loc = peak_points[key]["Peak"]
            pre_loc = peak_points[key]["Pre_Peak"]

            slope, intercept, r_value, p_value, std_err = linregress([pre_loc,peak_loc],[data[pre_loc],data[peak_loc]])
            upslopes.append(slope)

            post_loc = peak_points[key]["Post_Peak"]
            
            slope, intercept, r_value, p_value, std_err = linregress([peak_loc,post_loc],[data[peak_loc],data[post_loc]])
            downslopes.append(slope)

        if visualise == 1:
            plt.subplot(2,1,1)
            plt.title("Upslopes")
            plt.plot(data)
            for key in peak_points:
                peak = peak_points[key]["Peak"]
                pre = peak_points[key]["Pre_Peak"]
                plt.annotate(text = "", xy=(pre,data[pre]), xytext=(peak,data[peak]), arrowprops=dict(arrowstyle='<->'))
            plt.subplot(2,1,2)
            plt.title("Downslopes")
            plt.plot(data)
            for key in peak_points:
                peak = peak_points[key]["Peak"]
                post = peak_points[key]["Post_Peak"]
                plt.annotate(text = "", xy=(peak,data[peak]), xytext=(post,data[post]), arrowprops=dict(arrowstyle='<->'))
            #plt.axis('off')
            plt.show()

        return float(np.nanmedian(upslopes)), float(np.nanmedian(downslopes))
    else:
        return np.NaN, np.NaN