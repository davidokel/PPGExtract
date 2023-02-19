import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp
import data_methods
from scipy.integrate import trapz
    
def get_aucs(data,fs,visualise=0, debug = 0): 
    data = data_methods.normalise_data(data, 100)

    peak_points, peaks, troughs = data_methods.get_onsets_v2(data,100,debug=False)

    data_scaled = data_methods.data_scaler(data) # Scale the data to be between 0 and 1 (Used for plotting)

    auc = []
    sys_auc = []
    dia_auc = []
    auc_ratios = []

    if len(peaks) != 0:
        #peak_points = data_methods.get_onsets(data, peaks) # Given the data and the peak locations, get the onset locations
        
        # Printing and plotting used for debugging
        if debug == 1:
            # Get peaks, pre_peaks and post_peaks for all keys in the peak_points dictionary into 3 lists
            peak_list = []
            pre_peak_list = []
            post_peak_list = []

            for key in peak_points:
                pre_peak_list.append(peak_points[key]["Pre_peak"])
                post_peak_list.append(peak_points[key]["Post_peak"])
                peak_list.append(peak_points[key]["Peak"])
            
            # Print the len of the peak_list, pre_peak_list and post_peak_list
            print("AUC FUNCTION")
            print("Length of peak_list: ", len(peak_list))
            print("Length of pre_peak_list: ", len(pre_peak_list))
            print("Length of post_peak_list: ", len(post_peak_list))

            trough_list = pre_peak_list + post_peak_list

        # Iterate over the keys in the peak_points dictionary
        for key in peak_points:
            # For the current key, get the pre_peak, post_peak and peak locations
            pre = peak_points[key]["Pre_peak"]
            post = peak_points[key]["Post_peak"]
            peak_loc = peak_points[key]["Peak"]
            
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
                pre = peak_points[key]["Pre_peak"]
                post = peak_points[key]["Post_peak"]
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
                onset = peak_points[key]["Pre_peak"]
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
                onset = peak_points[key]["Post_peak"]
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
