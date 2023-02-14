import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp
import data_methods
from scipy.stats import linregress
from scipy.integrate import trapz
    
def get_aucs(data,fs,visualise=0, debug = 0):
    # Calling the get_peaks function from data_methods.py to find the peaks in the data
    peaks = data_methods.get_peaks(data, fs) # Given the data and the sampling frequency, get the peak locations
    data_scaled = data_methods.data_scaler(data) # Scale the data to be between 0 and 1 (Used for plotting)

    auc = []
    sys_auc = []
    dia_auc = []
    auc_ratios = []

    if len(peaks) != 0:
        peak_points = data_methods.get_onsets(data, peaks) # Given the data and the peak locations, get the onset locations
        
        # Printing and plotting used for debugging
        if debug == 1:
            plt.title("AUC and AUC ratios")
            plt.plot(data)
            plt.plot(peaks, data[peaks], "x")
            plt.show()

        for key in peak_points:
            peak_loc = peak_points[key]["Peak"]
            pre = peak_points[key]["Pre_Peak"]
            post = peak_points[key]["Post_Peak"]

            x = range(pre,post)
            y = []
            for index in x:
                y.append(abs(data[index]))
            auc.append(trapz(y,x))

            x = range(pre,peak_loc)
            y = []
            for index in x:
                y.append(abs(data[index]))
            sys_auc.append(trapz(y,x))

            x = range(peak_loc,post)
            y = []
            for index in x:
                y.append(abs(data[index]))
            dia_auc.append(trapz(y,x))

        for area in range(len(dia_auc)):
            auc_ratio = sys_auc[area]/dia_auc[area]
            auc_ratios.append(auc_ratio)

        if visualise == 1:
            plt.subplot(3,1,1)
            plt.title("Area under the curve (AUC)")
            plt.plot(data_scaled)
            for key in peak_points:
                pre = peak_points[key]["Pre_Peak"]
                post = peak_points[key]["Post_Peak"]
                x = range(pre,post)
                y = []
                for index in x:
                    y.append(abs(data_scaled[index]))
                plt.fill_between(x,y)
            #plt.axis('off')
            plt.subplot(3,1,2)
            plt.title("Systolic under the curve (S-AUC)")
            plt.plot(data_scaled)
            for key in peak_points:
                peak = peak_points[key]["Peak"]
                onset = peak_points[key]["Pre_Peak"]
                x = range(onset,peak)
                y = []
                for index in x:
                    y.append(abs(data_scaled[index]))
                plt.fill_between(x,y)
            #plt.axis('off')

            plt.subplot(3,1,3)
            plt.title("Diastolic under the curve (D-AUC)")
            plt.plot(data_scaled)
            for key in peak_points:
                peak = peak_points[key]["Peak"]
                onset = peak_points[key]["Post_Peak"]
                x = range(peak,onset)
                y = []
                for index in x:
                    y.append(abs(data_scaled[index]))
                plt.fill_between(x,y)

            plt.subplots_adjust(hspace=0.3)
            manager = plt.get_current_fig_manager()
            manager.window.showMaximized()
            #plt.axis('off')
            plt.axis('tight')
            plt.tight_layout()
            plt.show()

        return float(np.nanmedian(auc)), float(np.nanmedian(sys_auc)), float(np.nanmedian(dia_auc)), float(np.nanmedian(auc_ratios))
    else:
        return np.NaN, np.NaN, np.NaN, np.NaN
