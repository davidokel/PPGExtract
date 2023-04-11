import numpy as np
import matplotlib.pyplot as plt

def get_peak_times(pulse_data, visualise=0):
    data = pulse_data["norm_pulse_data"]
    peak = pulse_data["Relative_peak"]
    pre = 0
    post = len(pulse_data["norm_pulse_data"])

    rise_times = []
    decay_times = []

    if peak:
        rise_time = abs(peak-pre)
        rise_times.append(rise_time)

        decay_time = abs(post-peak)
        decay_times.append(decay_time)

        if visualise == 1:
            plt.subplot(2,1,1)
            plt.title("Rise times")
            plt.plot(data)
            plt.annotate(text = "", xy=(peak,data[pre]), xytext=(peak,data[peak]), arrowprops=dict(arrowstyle='-'))
            plt.annotate(text = "", xy=(pre,data[pre]), xytext=(peak,data[pre]), arrowprops=dict(arrowstyle='<->'))
            #plt.axis('off')
            
            plt.subplot(2,1,2)
            plt.title("Decay times")
            plt.plot(data)
            plt.annotate(text = "", xy=(peak,data[-1]), xytext=(peak,data[peak]), arrowprops=dict(arrowstyle='-'))
            plt.annotate(text = "", xy=(post,data[-1]), xytext=(peak,data[-1]), arrowprops=dict(arrowstyle='<->'))
        
            manager = plt.get_current_fig_manager()
            manager.window.showMaximized()
            #plt.axis('off')
            plt.show()

        return float(np.nanmedian(rise_times)), float(np.nanmedian(decay_times))
    else:
        return np.NaN, np.NaN