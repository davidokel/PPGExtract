import numpy as np
import scipy.stats as stats
import scipy.signal as sp
import matplotlib.pyplot as plt

def get_sqis(pulse_dictionary, fs, debug = False):

    for key in pulse_dictionary.keys():
        raw_pulse_data = pulse_dictionary[key]["raw_pulse_data"]
        norm_pulse_data = pulse_dictionary[key]["norm_pulse_data"]

        skew = get_skew(norm_pulse_data)
        kurt = get_kurt(norm_pulse_data)
        snr = get_snr(norm_pulse_data)
        zcr = get_zcr(norm_pulse_data,fs)
        ent = get_entropy(norm_pulse_data)
        pi = get_pi(raw_pulse_data, fs)

        # Create a dictionary with the sqis
        pulse_dictionary[key]["norm_mean"] = np.mean(norm_pulse_data)
        pulse_dictionary[key]["norm_median"] = np.median(norm_pulse_data)
        pulse_dictionary[key]["norm_variance"] = np.var(norm_pulse_data)
        pulse_dictionary[key]["raw_mean"] = np.mean(raw_pulse_data)
        pulse_dictionary[key]["raw_median"] = np.median(raw_pulse_data)
        pulse_dictionary[key]["raw_variance"] = np.var(raw_pulse_data)
        pulse_dictionary[key]["secder_norm_mean"] = np.mean(np.diff(np.diff(norm_pulse_data)))
        pulse_dictionary[key]["secder_norm_median"] = np.median(np.diff(np.diff(norm_pulse_data)))
        pulse_dictionary[key]["secder_norm_variance"] = np.var(np.diff(np.diff(norm_pulse_data)))
        pulse_dictionary[key]["skew"] = skew
        pulse_dictionary[key]["kurt"] = kurt
        pulse_dictionary[key]["snr"] = snr
        pulse_dictionary[key]["zcr"] = zcr
        pulse_dictionary[key]["ent"] = ent
        pulse_dictionary[key]["pi"] = pi
        pulse_dictionary[key]["class"] = "good"

        if debug:
            plt.subplot(2,1,1)
            plt.title("Raw Pulse Data")
            plt.plot(raw_pulse_data)
            plt.subplot(2,1,2)
            plt.title("Normalised Pulse Data")
            plt.plot(norm_pulse_data)
            plt.show()

    return pulse_dictionary

def get_skew(data):
    return stats.skew(data)

def get_kurt(data):
    return stats.kurtosis(data)

def get_snr(pulse):
    snr = np.std(np.abs(pulse)) / np.std(pulse)
    return snr

def get_zcr(pulse,fs):
    # Count the number of times the pulse crosses the zero-axis
    crossings = np.count_nonzero(np.diff(np.sign(pulse)) != 0)

    # Calculate the zero crossing rate (ZCR) in Hz
    zcr = crossings / (2 * len(pulse) / fs)

    return float(zcr)

def get_entropy(data):
    squared_signal = np.square(data)
    log_squared_signal = np.log(squared_signal)
    entropy = -np.sum(squared_signal * log_squared_signal)

    return entropy

def get_pi(raw_pulse, fs):
    raw_pulse = -1 * np.array(raw_pulse)
    
    sos_ac = sp.butter(2, [0.5, 12], btype='bandpass', analog=False, output='sos', fs=fs)
    sos_dc = sp.butter(4, (0.2/(fs/2)), btype='lowpass', analog=False, output='sos', fs=fs)
    try:
        ac_data = sp.sosfiltfilt(sos_ac, raw_pulse, axis=-1, padtype='odd', padlen=None)
        dc_data = sp.sosfiltfilt(sos_dc, raw_pulse, axis= -1, padtype='odd', padlen=None)
    except ValueError:
        min_padlen = int((len(raw_pulse) - 1) // 2)
        ac_data = sp.sosfiltfilt(sos_ac, raw_pulse, axis=-1, padtype='odd', padlen=min_padlen)
        dc_data = sp.sosfiltfilt(sos_dc, raw_pulse, axis= -1, padtype='odd', padlen=min_padlen)

    ac_component = max(ac_data) - min(ac_data)
    dc_component = np.mean(np.abs(dc_data))

    # Calculate the perfusion index
    pi = (ac_component / dc_component) * 100

    return pi