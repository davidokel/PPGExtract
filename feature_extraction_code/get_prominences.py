import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp
import support_code.data_methods as data_methods

def get_prominences(pulse_data, visualise=0):
    data = pulse_data["raw_pulse_data"]
    peak = np.array([pulse_data["Relative_peak"]])

    # Forcing positive values by adding 1
    data = 1 + data

    # Checking if there are any peaks in the data
    if peak:

        # Calculating the prominences and half_widths of the peaks
        prominences = sp.peak_prominences(data, peak)[0]
        contour_heights = data[peak] - prominences # Calculating the contour heights (Used for plotting)

        # Visualising the extracted features
        if visualise == 1:
            fig = plt.figure()
            plt.title("Amplitudes")
            plt.plot(data)
            plt.plot(peak,data[peak],'x')
            plt.vlines(x=peak, ymin=contour_heights, ymax=data[peak], label="amplitudes", linestyles="dashed", colors="green")
            plt.legend(loc="upper left")
            manager = plt.get_current_fig_manager()
            manager.window.showMaximized()
            #plt.axis('off')
            plt.show()
    
        return float(np.nanmedian(prominences))
    else:
        return np.NaN