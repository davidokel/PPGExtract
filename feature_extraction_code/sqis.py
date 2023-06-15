import numpy as np
import scipy.stats as stats
import scipy.signal as sp
import matplotlib.pyplot as plt

def get_sqis(pulse_dictionary, fs, visualise = False, debug = False):

    ##############################################################################
    # Initialising a list dictionary and lists to store statistical and sqi data #
    ##############################################################################
    sqi_dictionary = {}
    norm_means, norm_medians, norm_variances, raw_means, raw_medians, raw_variances, secder_norm_means, secder_norm_medians, secder_norm_variances, skews, kurts, snrs, zcrs, ents, pis = [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

    for key in pulse_dictionary.keys():
        raw_pulse_data = pulse_dictionary[key]["raw_pulse_data"]
        norm_pulse_data = pulse_dictionary[key]["norm_pulse_data"]

        skew = get_skew(norm_pulse_data)
        kurt = get_kurt(norm_pulse_data)
        snr = get_snr(norm_pulse_data)
        zcr = get_zcr(norm_pulse_data,fs)
        ent = get_entropy(norm_pulse_data)
        pi = get_pi(raw_pulse_data, fs)

        if visualise:
            plt.subplot(2,1,1)
            plt.title("Raw Pulse Data")
            plt.plot(raw_pulse_data)
            plt.subplot(2,1,2)
            plt.title("Normalised Pulse Data")
            plt.plot(norm_pulse_data)
            plt.show()

        # Appending the statistical/sqi data to the lists
        norm_means.append(np.mean(norm_pulse_data))
        norm_medians.append(np.median(norm_pulse_data))
        norm_variances.append(np.var(norm_pulse_data))
        raw_means.append(np.mean(raw_pulse_data))
        raw_medians.append(np.median(raw_pulse_data))
        raw_variances.append(np.var(raw_pulse_data))
        secder_norm_means.append(np.mean(np.gradient(np.gradient(norm_pulse_data))))
        secder_norm_medians.append(np.median(np.gradient(np.gradient(norm_pulse_data))))
        secder_norm_variances.append(np.var(np.gradient(np.gradient(norm_pulse_data))))
        skews.append(skew)
        kurts.append(kurt)
        snrs.append(snr)
        zcrs.append(zcr)
        ents.append(ent)
        pis.append(pi)
        
    # Appending the statistical/sqi data to the dictionary but calculating the nanmedian prior to adding
    sqi_dictionary["norm_mean"] = np.nanmedian(norm_means)
    sqi_dictionary["norm_median"] = np.nanmedian(norm_medians)
    sqi_dictionary["norm_variance"] = np.nanmedian(norm_variances)
    sqi_dictionary["raw_mean"] = np.nanmedian(raw_means)
    sqi_dictionary["raw_median"] = np.nanmedian(raw_medians)
    sqi_dictionary["raw_variance"] = np.nanmedian(raw_variances)
    sqi_dictionary["secder_norm_mean"] = np.nanmedian(secder_norm_means)
    sqi_dictionary["secder_norm_median"] = np.nanmedian(secder_norm_medians)
    sqi_dictionary["secder_norm_variance"] = np.nanmedian(secder_norm_variances)
    sqi_dictionary["skew"] = np.nanmedian(skews)
    sqi_dictionary["kurt"] = np.nanmedian(kurts)
    sqi_dictionary["snr"] = np.nanmedian(snrs)
    sqi_dictionary["zcr"] = np.nanmedian(zcrs)
    sqi_dictionary["ent"] = np.nanmedian(ents)
    sqi_dictionary["pi"] = np.nanmedian(pis)

    if debug:
        # Plotting the length of each list and its calculated nanmedian
        print("Length of norm_means: ", len(norm_means), " Calculated nanmedian: ", np.nanmedian(norm_means))
        print("Length of norm_medians: ", len(norm_medians), " Calculated nanmedian: ", np.nanmedian(norm_medians))
        print("Length of norm_variances: ", len(norm_variances), " Calculated nanmedian: ", np.nanmedian(norm_variances))
        print("Length of raw_means: ", len(raw_means), " Calculated nanmedian: ", np.nanmedian(raw_means))
        print("Length of raw_medians: ", len(raw_medians), " Calculated nanmedian: ", np.nanmedian(raw_medians))
        print("Length of raw_variances: ", len(raw_variances), " Calculated nanmedian: ", np.nanmedian(raw_variances))
        print("Length of secder_norm_means: ", len(secder_norm_means), " Calculated nanmedian: ", np.nanmedian(secder_norm_means))
        print("Length of secder_norm_medians: ", len(secder_norm_medians), " Calculated nanmedian: ", np.nanmedian(secder_norm_medians))
        print("Length of secder_norm_variances: ", len(secder_norm_variances), " Calculated nanmedian: ", np.nanmedian(secder_norm_variances))
        print("Length of skews: ", len(skews), " Calculated nanmedian: ", np.nanmedian(skews))
        print("Length of kurts: ", len(kurts), " Calculated nanmedian: ", np.nanmedian(kurts))
        print("Length of snrs: ", len(snrs), " Calculated nanmedian: ", np.nanmedian(snrs))
        print("Length of zcr: ", len(zcrs), " Calculated nanmedian: ", np.nanmedian(zcrs))
        print("Length of ent: ", len(ents), " Calculated nanmedian: ", np.nanmedian(ents))
        print("Length of pi: ", len(pis), " Calculated nanmedian: ", np.nanmedian(pis))
        
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