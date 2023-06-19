from feature_extraction_code.get_aucs import get_aucs
from feature_extraction_code.get_widths import get_widths
from feature_extraction_code.get_peak_times import get_peak_times
from feature_extraction_code.get_prominences import get_prominences
from feature_extraction_code.get_second_der_ratio import get_second_der_ratio
from feature_extraction_code.get_slopes import get_slopes
from feature_extraction_code.get_beat_based_features import get_beat_features
from feature_extraction_code.get_datum_line_features import get_datum_line_features
from feature_extraction_code.sqis import get_sqis

def get_features_v2(raw_data, pulse_dictionary, visualise=False, debug=False):
        """auc, sys_auc, dia_auc, auc_ratio = get_aucs(pulse_dictionary, visualise, debug)
        rise_time, decay_time, rise_decay_time_ratio = get_peak_times(pulse_dictionary, visualise)
        num_beats, median_ibi, std_ibi, cv_ibi = get_beat_features(pulse_dictionary, debug)
        prominence = get_prominences(pulse_dictionary, visualise)
        sec_der_ratio = get_second_der_ratio(pulse_dictionary)
        slope_features = get_slopes(pulse_dictionary, visualise)
        datum_features = get_datum_line_features(pulse_dictionary, visualise, debug)
        width_dictionary = get_widths(pulse_dictionary, visualise, debug)"""
        sqi_dictionary = get_sqis(pulse_dictionary, visualise=visualise, debug=debug)

        return pulse_dictionary
