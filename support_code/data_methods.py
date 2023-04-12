import numpy as np
import scipy.signal as sp

def normalise_data(data,fs):
    sos_ac = sp.butter(2, [0.5, 12], btype='bandpass', analog=False, output='sos', fs=fs)
    sos_dc = sp.butter(4, (0.2/(fs/2)), btype='lowpass', analog=False, output='sos', fs=fs)
    try:
        ac = sp.sosfiltfilt(sos_ac, data, axis=-1, padtype='odd', padlen=None)
        dc = sp.sosfiltfilt(sos_dc, data, axis= -1, padtype='odd', padlen=None)
    except ValueError:
        min_padlen = int((len(data) - 1) // 2)
        ac = sp.sosfiltfilt(sos_ac, data, axis=-1, padtype='odd', padlen=min_padlen)
        dc = sp.sosfiltfilt(sos_dc, data, axis= -1, padtype='odd', padlen=min_padlen)
    
    # Normalising and scaling the signal to a more manageable range
    normalised = 10*(-(ac/dc))

    return normalised

def band_pass_filter(data, order, fs, low_cut, high_cut):
    sos = sp.butter(order, [low_cut, high_cut], btype='bandpass', analog=False, output='sos', fs=fs) # Defining a high pass filter
    filtered_data = sp.sosfiltfilt(sos, data, axis=- 1, padtype='odd', padlen=None) # Applying the high pass filter to the data chunk
    return filtered_data

def data_scaler(data):
    min_data = min(data)
    if min_data < 0:
        scaling_factor = abs(min_data)
    else:
        scaling_factor = np.diff([0, min_data])[0]

    data = data + scaling_factor
    return data

def get_signal_slopes(data,index1,index2):
    # Compute the difference in y-values and x-values
    y_diff = data[index2] - data[index1]
    x_diff = index2 - index1
    
    # Compute the slope
    slope = y_diff / x_diff
    
    return slope