import numpy as np
import scipy.stats as stats
import scipy.signal as sp
import matplotlib.pyplot as plt
import data_methods

def get_sqis(data, fs, visualise = 0):
    # Call all sqi functions

    # Isolating the AC component of the data
    sos_ac = sp.butter(2, [0.5, 12], btype='bandpass', analog=False, output='sos', fs=fs) # Defining a high pass filter
    ac_data = sp.sosfiltfilt(sos_ac, data, axis=- 1, padtype='odd', padlen=None) # Applying the high pass filter to the data chunk
    
    skew, kurt = sqi_statistics(ac_data)
    snr = sqi_snr(ac_data)
    zcr = sqi_zcr(ac_data,fs)
    ent = sqi_entropy(ac_data)
    pi = sqi_pi(data, fs)

    # Create a dictionary with the sqis
    sqi_dictionary = {'skew': skew, 'kurt': kurt, 'snr': snr, 'zcr': zcr, 'ent': ent, 'pi': pi}

    if visualise == 1:
        print('Skewness: ', skew)
        print('Kurtosis: ', kurt)
        print('SNR: ', snr)
        print('ZCR: ', zcr)
        print('Entropy: ', ent)
        print('PI: ', pi)

    return skew, kurt, snr, zcr, ent, pi, sqi_dictionary

def sqi_statistics(data):
    """ 
    Estimates the skewness amd kurtosis given data
    
    Inputs:     data, the data to be analysed
    Outputs:    skew, value of the skewness
                kurt, value of the kurtosis
    """
    skew = stats.skew(data)
    kurt = stats.kurtosis(data)
    
    return float(skew), float(kurt)

def sqi_snr(data, fs = 100):
    """data = np.asanyarray(data)
    m = data.mean(axis)
    sd = data.std(axis=axis, ddof=ddof)

    #return np.where(sd == 0, 0, m/sd)
    return float(20*np.log10(abs(np.where(sd == 0, 0, m/sd))))"""

    sos = sp.butter(10, 10, btype = 'highpass', analog = False, output = 'sos', fs = fs)
    noise = sp.sosfiltfilt(sos, data)
    
    snr = 10*np.log10(np.sum(data)**2/np.sum(noise)**2)
    
    return snr

def sqi_zcr (data,fs):
    """
    What this is doing:

    - creates a boolean array of where the signal is above 0 (audioData > 0)
    - does a pairwise difference (np.diff) so locations of zero crossings become 1 (rising) and -1 (falling)
    - picks the index of the array where those nonzero values are (np.nonzero).

    Then if you want the number of crossings, you can just take zero_crosses.size.
    """
    zero_crosses = np.nonzero(np.diff(data > 0))[0]
    zcr = zero_crosses.size
    
    return float(zcr)

def sqi_entropy (data):
    data = (data - np.min(data))/(np.max(data) - np.min(data))
    ent = stats.entropy(data)
    
    return float(ent)

def sqi_pi (data, fs):
    """
    PI is an indicator of the relative strength of the pulsatile signal from pulse oximetry 
    and has been found to be a reliable indicator of peripheral perfusion. 
    PI is calculated by dividing the pulsatile signal (AC) by the nonpulsatile signal (DC) times 100, 
    and is expressed as a percent ranging from 0.02% to 20%. A higher PI value, therefore, 
    indicates a stronger pulsatile signal and better peripheral circulation at the sensor site. 
    """

    sos_ac = sp.butter(2, [0.5, 12], btype='bandpass', analog=False, output='sos', fs=fs) # Defining a high pass filter
    ac_data = sp.sosfiltfilt(sos_ac, data, axis=- 1, padtype='odd', padlen=None) # Applying the high pass filter to the data chunk

    # check that ac_data is not size 0
    if len(ac_data) > 0:
        upper_envelope, lower_envelope, peaks, troughs, envelope_difference = data_methods.get_envelope(ac_data, 2, 100)
    
    # check if envelope_differeeence is empty
    if len(envelope_difference) <= 0:
        pi = 0
    else:
        # Element wise division of the envelope_dfference and the dc_data
        pi = envelope_difference/abs(np.mean(data))
        pi = np.mean(pi)*100

    return float(pi)
