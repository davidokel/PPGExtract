from feature_extraction_code.get_aucs import get_aucs
from feature_extraction_code.get_half_widths import get_half_widths
from feature_extraction_code.get_peak_times import get_peak_times
from feature_extraction_code.get_prominences import get_prominences
from feature_extraction_code.get_second_der_ratio import get_second_der_ratio
from feature_extraction_code.get_slopes import get_slopes

def get_features(pulse_dictionary, visualise=False):
    for key in pulse_dictionary.keys():
            pulse_data = pulse_dictionary[key]
            
            auc, sys_auc, dia_auc, auc_ratio = get_aucs(pulse_data, visualise)
            half_width = get_half_widths(pulse_data, visualise)
            rise_time, decay_time = get_peak_times(pulse_data, visualise)
            prominence = get_prominences(pulse_data, visualise)
            sec_der_ratio = get_second_der_ratio(pulse_data)
            slope_features = get_slopes(pulse_data, visualise)

            features = {"auc": auc, "sys_auc": sys_auc, "dia_auc": dia_auc, "auc_ratio": auc_ratio, "half_width": half_width, "rise_time": rise_time, "decay_time": decay_time, "prominence": prominence, "sec_der_ratio": sec_der_ratio}
            features.update(slope_features)
            
            pulse_dictionary[key]["features"] = features
    
    return pulse_dictionary