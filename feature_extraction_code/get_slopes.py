import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

def get_slopes(pulse_data, visualise=0):
    data = pulse_data["raw_pulse_data"]
    peak = pulse_data["Relative_peak"]
    pre = 0
    post = len(pulse_data["norm_pulse_data"])

    if peak:
        # Create an empty dictionary to store the features
        features = {}

        peak_height = data[peak] - data[pre]

        # Using the distance formula to find the straigh line length between the onset and peak and peak and end of the pulse
        # distance = sqrt((x2 - x1)^2 + (y2 - y1)^2)
        upslope_length = np.sqrt((peak - pre)**2 + (data[peak] - data[pre])**2)
        downslope_length = np.sqrt((post - peak)**2 + (data[-1] - data[peak])**2)
        features['upslope_length'] = upslope_length
        features['downslope_length'] = downslope_length

        # UPSLOPE
        # The upslope measures the rate of change of the pulse during the rising phase, from the onset to the peak. 
        # This can give an indication of how quickly the pulse is rising and how steep the rising edge is.
        upslope, intercept, r_value, p_value, std_err = linregress([pre,peak],[data[pre],data[peak]])
        features['upslope'] = upslope

        # DOWNSLOPE
        # The downslope measures the rate of change of the pulse during the falling phase, from the peak to the end of the pulse. 
        # This can give an indication of how quickly the pulse is falling and how steep the falling edge is.
        downslope, intercept, r_value, p_value, std_err = linregress([peak,post],[data[peak],data[-1]])
        features['downslope'] = downslope

        # ONSET_END_SLOPE
        # The onset-end slope measures the rate of change of the pulse over the entire pulse length, from the onset to the end of the pulse. 
        # This can give an indication of the overall shape of the pulse and how steep it is.
        onset_end_slope, intercept, r_value, p_value, std_err = linregress([pre, post], [data[pre], data[-1]])
        features['onset_end_slope'] = onset_end_slope

        # UPSLOPE_DOWNSLOPE_RATIO
        # This ratio can give an indication of how steep the pulse is on the rising edge (upslope) compared to the falling edge (downslope). 
        # If the ratio is greater than 1, then the pulse rises more steeply than it falls, and if the ratio is less than 1, then the pulse falls more steeply than it rises. 
        # This could be useful for comparing different pulses to see which ones have a more pronounced rising or falling edge.
        upslope_downslope_ratio = upslope / downslope if downslope != 0 else np.nan
        features['upslope_downslope_ratio'] = upslope_downslope_ratio

        # PULSE_LENGTH_HEIGHT_RATIO
        # This ratio can give an indication of the overall shape of the pulse, specifically how long it takes to return to baseline after reaching its peak.
        # If the ratio is high, then the pulse takes a relatively long time to return to baseline after reaching its peak, indicating a broader shape. 
        # If the ratio is low, then the pulse returns quickly to baseline, indicating a narrower shape. 
        # This could be useful for comparing different pulses to see if there are consistent differences in the shape of the pulse.
        pulse_length_height_ratio = (post - pre) / (data[peak] - data[pre]) if data[peak] != data[pre] else np.nan
        features['pulse_length_height_ratio'] = pulse_length_height_ratio

        # UPSLOPE_DONWSLOPE_LENGTH_RATIO
        # This ratio can help to quantify the relative contribution of the rising phase of the pulse to the overall length of the pulse. 
        # A pulse with a higher ratio would have a longer rising phase relative to its overall length, while a pulse with a lower ratio would have a shorter rising phase. 
        upslope_downslope_length_ratio = upslope_length / downslope_length if downslope_length != 0 else np.nan
        features['upslope_downslope_length_ratio'] = upslope_downslope_length_ratio

        # UPSLOPE_PULSE_LENGTH_RATIO
        # This ratio measures the proportion of the total pulse length that is made up of the rising phase. 
        # This can be useful in characterizing the slope of the rising phase and how it contributes to the overall shape of the pulse. 
        # For example, a pulse with a higher ratio may have a steeper or more pronounced rise, while a pulse with a lower ratio may have a more gradual rise.
        upslope_pulse_length_ratio = upslope_length / (post - pre) if (post - pre) != 0 else np.nan
        features['upslope_pulse_length_ratio'] = upslope_pulse_length_ratio

        # PULSE_DOWNSLOPE_LENGTH_RATIO
        # This ratio measures the proportion of the total pulse length that is made up of the falling phase, from the peak to the end of the pulse. 
        # This can be useful in characterizing the slope of the falling phase and how it contributes to the overall shape of the pulse. 
        # For example, a pulse with a higher ratio may have a steeper or more pronounced fall, while a pulse with a lower ratio may have a more gradual fall.
        pulse_downslope_length_ratio = downslope_length / (post - pre) if (post - pre) != 0 else np.nan
        features['pulse_downslope_length_ratio'] = pulse_downslope_length_ratio

        # HEIGHT_UPSLOPE_RATIO
        # The ratio of the height of the pulse relative to the length of the rising phase.
        # This ratio provides information about how steep the rising phase of the pulse is in relation to its height. 
        # A steeper slope will result in a higher ratio, while a shallower slope will result in a lower ratio.
        height_upslope_ratio = peak_height / upslope_length if upslope_length != 0 else np.nan
        features['height_upslope_ratio'] = height_upslope_ratio

        # HEIGHT_DOWNSLOPE_RATIO
        # The ratio of the height of the pulse relative to the length of the falling phase.
        # This ratio provides information about how steep the falling phase of the pulse is in relation to its height. 
        # A steeper slope will result in a higher ratio, while a shallower slope will result in a lower ratio.
        height_downslope_ratio = peak_height / downslope_length if downslope_length != 0 else np.nan
        features['height_downslope_ratio'] = height_downslope_ratio

        if visualise == 1:
            plt.subplot(2,1,1)
            plt.title("Upslopes")
            plt.plot(data)
            plt.annotate(text = "", xy=(pre,data[pre]), xytext=(peak,data[peak]), arrowprops=dict(arrowstyle='<->'))
            
            plt.subplot(2,1,2)
            plt.title("Downslopes")
            plt.plot(data)
            plt.annotate(text = "", xy=(peak,data[peak]), xytext=(post,data[-1]), arrowprops=dict(arrowstyle='<->'))
            #plt.axis('off')
            plt.show()

        return features 
    else:
        return np.NaN, np.NaN