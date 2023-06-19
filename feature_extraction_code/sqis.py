import numpy as np
import scipy.stats as stats
import scipy.signal as sp
import matplotlib.pyplot as plt
from support_code.data_methods import data_scaler 

def get_sqis(pulse_dictionary, visualise = False, debug = False):
    """
    Calculates statistical and SQI data for each pulse in the pulse dictionary.

    Args:
        pulse_dictionary (dict): Dictionary containing pulse data and peak information.
        fs (float): Sampling frequency of the pulse data.
        visualise (bool, optional): Whether to visualise the statistical and SQI data. Defaults to False.
        debug (bool, optional): Whether to print debug information. Defaults to False.

    Returns:
        dict: Dictionary containing statistical and SQI data for each pulse.

    Raises:
        ValueError: If the pulse_dictionary is empty.
    """

    ##############################################################################
    # Initialising a list dictionary and lists to store statistical and sqi data #
    ##############################################################################
    sqi_dictionary = {}
    raw_means, raw_medians, raw_variances, skews, kurts, snrs, zcrs, ents, pis = [], [], [], [], [], [], [], [], []

    for key in pulse_dictionary.keys():
        pulse_data = pulse_dictionary[key]["pulse_data"]
        raw_pulse_data = pulse_dictionary[key]["raw_pulse_data"]
        peak = pulse_dictionary[key]["relative_peak"]

        skew = get_skew(pulse_data)
        kurt = get_kurt(pulse_data)
        snr = get_snr(raw_pulse_data, pulse_data)
        zcr = get_zcr(pulse_data)
        ent = get_entropy(pulse_data)
        pi = get_pi(pulse_data, peak)

        # Appending the statistical/sqi data to the lists
        raw_means.append(np.mean(pulse_data))
        raw_medians.append(np.median(pulse_data))
        raw_variances.append(np.var(pulse_data))
        skews.append(skew)
        kurts.append(kurt)
        snrs.append(snr)
        zcrs.append(zcr)
        ents.append(ent)
        pis.append(pi)

        #########
        # Debug #
        #########
        if debug:
            print("Raw mean: ", raw_means[-1])
            print("Raw median: ", raw_medians[-1])
            print("Raw variance: ", raw_variances[-1])
            print("Skew: ", skews[-1])
            print("Kurt: ", kurts[-1])
            print("SNR: ", snrs[-1])
            print("ZCR: ", zcrs[-1])
            print("Entropy: ", ents[-1])
            print("PI: ", pis[-1])

        ############
        # Plotting #
        ############
        if visualise:
            plt.title("Pulse Data")
            plt.plot(pulse_data)
            plt.show()

            # Get the user input to see if they want to move onto the next visualisation or stop visualising
            user_input = input("Press enter to continue, or type 'stop' to stop visualising: ")
            if user_input == "stop":
                visualise = 0

    # Appending the statistical/sqi data to the dictionary but calculating the nanmedian prior to adding
    sqi_dictionary["data_mean"] = np.nanmedian(raw_means)
    sqi_dictionary["data_median"] = np.nanmedian(raw_medians)
    sqi_dictionary["data_variance"] = np.nanmedian(raw_variances)
    sqi_dictionary["skew"] = np.nanmedian(skews)
    sqi_dictionary["kurt"] = np.nanmedian(kurts)
    sqi_dictionary["snr"] = np.nanmedian(snrs)
    sqi_dictionary["zcr"] = np.nanmedian(zcrs)
    sqi_dictionary["ent"] = np.nanmedian(ents)
    sqi_dictionary["pi"] = np.nanmedian(pis)

    #########
    # Debug #
    #########
    if debug:
        # Plotting the length of each list and its calculated nanmedian
        print("Length of raw_means: ", len(raw_means), " Calculated nanmedian: ", np.nanmedian(raw_means))
        print("Length of raw_medians: ", len(raw_medians), " Calculated nanmedian: ", np.nanmedian(raw_medians))
        print("Length of raw_variances: ", len(raw_variances), " Calculated nanmedian: ", np.nanmedian(raw_variances))
        print("Length of skews: ", len(skews), " Calculated nanmedian: ", np.nanmedian(skews))
        print("Length of kurts: ", len(kurts), " Calculated nanmedian: ", np.nanmedian(kurts))
        print("Length of snrs: ", len(snrs), " Calculated nanmedian: ", np.nanmedian(snrs))
        print("Length of zcr: ", len(zcrs), " Calculated nanmedian: ", np.nanmedian(zcrs))
        print("Length of ent: ", len(ents), " Calculated nanmedian: ", np.nanmedian(ents))
        print("Length of pi: ", len(pis), " Calculated nanmedian: ", np.nanmedian(pis))
        
    return pulse_dictionary

def get_skew(pulse_data):
    """
    Calculates the skewness of a given dataset.

    Args:
        pulse_data (array-like): The dataset for which the skewness is to be calculated.

    Returns:
        float: The skewness of the dataset.

    Raises:
        ValueError: If the data is empty.
    """
    return stats.skew(pulse_data)

def get_kurt(pulse_data):
    """
    Calculates the kurtosis of a given dataset.

    Args:
        pulse_data (array-like): The dataset for which the kurtosis is to be calculated.

    Returns:
        float: The kurtosis of the dataset.

    Raises:
        ValueError: If the data is empty.
    """
    return stats.kurtosis(pulse_data)

def get_snr(unfiltered_data, filtered_data):
    """
    Calculates the signal-to-noise ratio (SNR) of a given pulse.

    Args:
        pulse_data (array-like): The pulse data.

    Returns:
        float: The SNR of the pulse.
    """
    noise = unfiltered_data - filtered_data
    signal_power = np.var(filtered_data)
    noise_power = np.var(noise)
    snr = signal_power / noise_power
    return snr

def get_zcr(pulse_data):
    """
    Calculates the zero crossing rate (ZCR) of a given pulse.

    Args:
        pulse_data (array-like): The pulse data for which the ZCR is to be calculated.
        fs (float): The sampling frequency of the pulse data.

    Returns:
        float: The ZCR of the pulse.

    Raises:
        ValueError: If the pulse_data is empty.
    """
        
    # Normalising the data by subtracting the mean
    normalised_pulse_data = pulse_data - np.mean(pulse_data)

    # Count zero crossings
    zcr = len(np.where(np.diff(np.signbit(normalised_pulse_data)))[0])

    # Calculate zero-crossing rate
    pulse_duration = len(normalised_pulse_data)
    zcr_rate = zcr / pulse_duration

    return zcr_rate

def get_entropy(pulse_data):
    """
    Calculates the entropy of a given dataset.

    Args:
        pulse_data (array-like): The dataset for which the entropy is to be calculated.

    Returns:
        float: The entropy of the dataset.

    Raises:
        ValueError: If the pulse_data is empty.
    """
    entropy = stats.entropy(pulse_data)

    return entropy

def get_pi(pulse_data, peak):
    """
    Calculates the PI (Peak-to-Instantaneous Ratio) value for a given pulse.

    Args:
        pulse_data (array-like): The pulse data.
        peak (int): The index of the relative peak in the pulse data.

    Returns:
        float: The PI value for the pulse.

    Raises:
        ValueError: If the pulse_data is empty or the peak index is out of range.
    """
    # Get the prominence of the pulse using the relative peak using peak_prominences
    peak_prominence = sp.peak_prominences(pulse_data, [peak])[0][0]

    # Get the mean of the pulse
    mean = np.mean(pulse_data)

    # Calculate the PI value
    pi = (abs(peak_prominence) / abs(mean)) * 100

    return pi