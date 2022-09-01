from turtle import color
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp
import math
from data_methods import get_peaks, get_onsets
from scipy.integrate import simps, trapz
from scipy.stats import linregress

def get_upslopes_downslopes_rise_times_auc(data,fs,visualise=0):
    peaks = get_peaks(data, fs)

    rise_times = []
    decay_times = []

    upslopes = []
    downslopes = []

    auc = []
    sys_auc = []
    dia_auc = []
    auc_ratios = []

    if len(peaks) != 0:
        peak_points = get_onsets(data, peaks)

        for key in peak_points:
            peak_loc = peak_points[key]["Peak"]
            pre_loc = peak_points[key]["Pre_Peak"]

            slope, intercept, r_value, p_value, std_err = linregress([pre_loc,peak_loc],[data[pre_loc],data[peak_loc]])
            
            rise_time = (peak_loc-pre_loc)
            rise_times.append(rise_time)
            upslopes.append(slope)

            post_loc = peak_points[key]["Post_Peak"]
            
            slope, intercept, r_value, p_value, std_err = linregress([peak_loc,post_loc],[data[peak_loc],data[post_loc]])

            decay_time = (post_loc-peak_loc)
            decay_times.append(decay_time)
            downslopes.append(slope)

        for key in peak_points:
            peak_loc = peak_points[key]["Peak"]
            pre = peak_points[key]["Pre_Peak"]
            post = peak_points[key]["Post_Peak"]

            x = range(pre,post)
            y = []
            for index in x:
                y.append(data[index])
            auc.append(trapz(y,x))

            x = range(pre,peak_loc)
            y = []
            for index in x:
                y.append(data[index])
            sys_auc.append(trapz(y,x))

            x = range(peak_loc,post)
            y = []
            for index in x:
                y.append(data[index])
            dia_auc.append(trapz(y,x))

        for area in range(len(dia_auc)):
            auc_ratio = dia_auc[area]/sys_auc[area]
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

            plt.subplot(4,1,2)
            plt.title("Downslopes")
            plt.plot(data)
            for key in peak_points:
                peak = peak_points[key]["Peak"]
                post = peak_points[key]["Post_Peak"]
                plt.annotate(text = "", xy=(peak,data[peak]), xytext=(post,data[post]), arrowprops=dict(arrowstyle='<->'))

            plt.subplot(4,1,3)
            plt.title("Rise times")
            plt.plot(data)
            for key in peak_points:
                peak = peak_points[key]["Peak"]
                pre = peak_points[key]["Pre_Peak"]

                plt.annotate(text = "", xy=(peak,data[pre]), xytext=(peak,data[peak]), arrowprops=dict(arrowstyle='-'))
                plt.annotate(text = "", xy=(pre,data[pre]), xytext=(peak,data[peak]), arrowprops=dict(arrowstyle='<->'))

            plt.subplot(4,1,4)
            plt.title("Decay times")
            plt.plot(data)
            for key in peak_points:
                peak = peak_points[key]["Peak"]
                post = peak_points[key]["Post_Peak"]

                plt.annotate(text = "", xy=(peak,data[post]), xytext=(peak,data[peak]), arrowprops=dict(arrowstyle='-'))
                plt.annotate(text = "", xy=(post,data[post]), xytext=(peak,data[peak]), arrowprops=dict(arrowstyle='<->'))
            
            manager = plt.get_current_fig_manager()
            manager.window.showMaximized()
            plt.show()

            plt.subplot(3,1,1)
            plt.title("Area under the curve (AUC)")
            plt.plot(data)
            for key in peak_points:
                pre = peak_points[key]["Pre_Peak"]
                post = peak_points[key]["Post_Peak"]
                x = range(pre,post)
                y = []
                for index in x:
                    y.append(data[index])
                plt.fill_between(x,y)
            
            plt.subplot(3,1,2)
            plt.title("Systolic under the curve (S-AUC)")
            plt.plot(data)
            for key in peak_points:
                peak = peak_points[key]["Peak"]
                onset = peak_points[key]["Pre_Peak"]
                x = range(onset,peak)
                y = []
                for index in x:
                    y.append(data[index])
                plt.fill_between(x,y)

            plt.subplot(3,1,3)
            plt.title("Diastolic under the curve (D-AUC)")
            plt.plot(data)
            for key in peak_points:
                peak = peak_points[key]["Peak"]
                onset = peak_points[key]["Post_Peak"]
                x = range(peak,onset)
                y = []
                for index in x:
                    y.append(data[index])
                plt.fill_between(x,y)
            
            manager = plt.get_current_fig_manager()
            manager.window.showMaximized()
            plt.show()
    return np.nanmedian(upslopes), np.nanmedian(downslopes), np.nanmedian(rise_times), np.nanmedian(decay_times), np.nanmedian(auc), np.nanmedian(sys_auc), np.nanmedian(dia_auc), np.nanmedian(auc_ratio), np.nanmedian(second_derivative_ratio)
