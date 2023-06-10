import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp
import support_code.data_methods as data_methods
from scipy.stats import linregress
from scipy.integrate import trapz

def get_half_widths(pulse_data, visualise=0):
    data = pulse_data["raw_pulse_data"]
    peak = np.array([pulse_data["Relative_peak"]])

    # Forcing positive values by adding 1
    data = 1 + data

    # Initialising lists for the prominences and half widths
    half_widths = []

    # Checking if there are any peaks in the data
    if peak:
        # Calculating the half and full widths of the peaks
        half_widths_res = sp.peak_widths(data, peak, rel_height=0.5) # Calculating the half widths of the peaks using the scipy function peak_widths
        half_widths = sp.peak_widths(data, peak, rel_height=0.5)[0].tolist()
        
        # Visualising the extracted features
        if visualise == 1:
            fig = plt.figure()
            plt.title("Half peak widths")
            plt.plot(data)
            plt.hlines(*half_widths_res[1:], color="C2", label="half peak widths")
            plt.legend(loc="upper left")

            manager = plt.get_current_fig_manager()
            manager.window.showMaximized()
            #plt.axis('off')
            plt.show()
    
        return float(np.nanmedian(half_widths))
    else:
        return np.NaN