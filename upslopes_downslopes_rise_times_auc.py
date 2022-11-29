import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapz
from scipy.stats import linregress
import data_methods

def get_upslopes_downslopes_rise_times_auc(data,fs,visualise=0, debug = 0):
    peaks = data_methods.get_peaks(data, fs) # Given the data and the sampling frequency, get the peak locations
    data_scaled = data_methods.data_scaler(data) # Scale the data to be between 0 and 1 (Used for plotting)

    rise_times = []
    decay_times = []

    upslopes = []
    downslopes = []

    auc = []
    sys_auc = []
    dia_auc = []
    auc_ratios = []

    second_derivative_ratio = 0

    seconds = int((len(data)/fs)*0.25) # Simplified quality assessment, if there are less peaks than 1/4 of the data length (in seconds) then don't extract features
    if (len(peaks) > seconds) == True:
        peak_points = data_methods.get_onsets(data, peaks) # Given the data and the peak locations, get the onset locations
        
        # Printing and plotting used for debugging
        if debug == 1:
            for key in peak_points:
                print("Peak: ", key)
                print("Onset: ", peak_points[key]["Pre_Peak"])
                print("Peak: ", peak_points[key]["Peak"])
                print("Offset: ", peak_points[key]["Post_Peak"])
                print("")

                # Plot peak, onset and offset
                plt.plot(data)
                plt.plot(peak_points[key]["Pre_Peak"], data[peak_points[key]["Pre_Peak"]], "x")
                plt.plot(peak_points[key]["Peak"], data[peak_points[key]["Peak"]], "x")
                plt.plot(peak_points[key]["Post_Peak"], data[peak_points[key]["Post_Peak"]], "x")
                plt.show()

        for key in peak_points:
            peak_loc = peak_points[key]["Peak"]
            pre_loc = peak_points[key]["Pre_Peak"]

            slope, intercept, r_value, p_value, std_err = linregress([pre_loc,peak_loc],[data[pre_loc],data[peak_loc]])
            
            rise_time = abs(peak_loc-pre_loc)
            rise_times.append(rise_time)
            upslopes.append(slope)

            post_loc = peak_points[key]["Post_Peak"]
            
            slope, intercept, r_value, p_value, std_err = linregress([peak_loc,post_loc],[data[peak_loc],data[post_loc]])

            decay_time = abs(post_loc-peak_loc)
            decay_times.append(decay_time)
            downslopes.append(slope)

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

        second_derivative = np.diff(np.diff(data))
        max_value = max(second_derivative)
        min_value = min(second_derivative)
        second_derivative_ratio = max_value/min_value

        if visualise == 1:
            plt.subplot(4,1,1)
            plt.title("Upslopes")
            plt.plot(data)
            for key in peak_points:
                peak = peak_points[key]["Peak"]
                pre = peak_points[key]["Pre_Peak"]
                plt.annotate(text = "", xy=(pre,data[pre]), xytext=(peak,data[peak]), arrowprops=dict(arrowstyle='<->'))
            plt.axis('off')
            plt.subplot(4,1,2)
            plt.title("Downslopes")
            plt.plot(data)
            for key in peak_points:
                peak = peak_points[key]["Peak"]
                post = peak_points[key]["Post_Peak"]
                plt.annotate(text = "", xy=(peak,data[peak]), xytext=(post,data[post]), arrowprops=dict(arrowstyle='<->'))
            plt.axis('off')
            plt.subplot(4,1,3)
            plt.title("Rise times")
            plt.plot(data)
            for key in peak_points:
                peak = peak_points[key]["Peak"]
                pre = peak_points[key]["Pre_Peak"]

                plt.annotate(text = "", xy=(peak,data[pre]), xytext=(peak,data[peak]), arrowprops=dict(arrowstyle='-'))
                plt.annotate(text = "", xy=(pre,data[pre]), xytext=(peak,data[pre]), arrowprops=dict(arrowstyle='<->'))
            plt.axis('off')
            plt.subplot(4,1,4)
            plt.title("Decay times")
            plt.plot(data)
            for key in peak_points:
                peak = peak_points[key]["Peak"]
                post = peak_points[key]["Post_Peak"]

                plt.annotate(text = "", xy=(peak,data[post]), xytext=(peak,data[peak]), arrowprops=dict(arrowstyle='-'))
                plt.annotate(text = "", xy=(post,data[post]), xytext=(peak,data[post]), arrowprops=dict(arrowstyle='<->'))
            
            manager = plt.get_current_fig_manager()
            manager.window.showMaximized()
            plt.axis('off')
            plt.axis('tight')
            plt.show()

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
            plt.axis('off')
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
            plt.axis('off')

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
            plt.axis('off')
            plt.axis('tight')
            plt.show()

            print("Upslope: ", float(np.nanmedian(upslopes)))
            print("Downslope: ", float(np.nanmedian(downslopes)))
            print("Rise time: ", float(np.nanmedian(rise_times)))
            print("Decay time: ", float(np.nanmedian(decay_times)))
            print("AUC: ", float(np.nanmedian(auc)))
            print("S-AUC: ", float(np.nanmedian(sys_auc)))
            print("D-AUC: ", float(np.nanmedian(dia_auc)))
            print("AUC ratio: ", float(np.nanmedian(auc_ratios)))
            print("Second derivative ratio: ", second_derivative_ratio)

        return float(np.nanmedian(upslopes)), float(np.nanmedian(downslopes)), float(np.nanmedian(rise_times)), float(np.nanmedian(decay_times)), float(np.nanmedian(auc)), float(np.nanmedian(sys_auc)), float(np.nanmedian(dia_auc)), float(np.nanmedian(auc_ratios)), float(abs(second_derivative_ratio))
    else:
        return np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN