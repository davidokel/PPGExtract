from feature_extraction_code.get_aucs import get_aucs
from feature_extraction_code.get_widths import get_widths
from feature_extraction_code.get_peak_times import get_peak_times
from feature_extraction_code.get_prominences import get_prominences
from feature_extraction_code.get_slopes import get_slopes
from feature_extraction_code.get_beat_based_features import get_beat_features
from feature_extraction_code.get_datum_line_features import get_datum_line_features
from feature_extraction_code.sqis import get_sqis

def get_features_v2(pulse_dictionary, fs, visualise=False, debug=False):
        auc_features = get_aucs(pulse_dictionary, visualise, debug)
        time_features = get_peak_times(pulse_dictionary, visualise)
        beat_features = get_beat_features(pulse_dictionary, debug)
        prominence_feature = get_prominences(pulse_dictionary, visualise)
        #sec_der_ratio = get_second_der_ratio(pulse_dictionary)
        slope_features = get_slopes(pulse_dictionary, visualise)
        datum_features = get_datum_line_features(pulse_dictionary, visualise, debug)
        width_features = get_widths(pulse_dictionary, visualise, debug)
        sqi_features = get_sqis(pulse_dictionary, fs, visualise=visualise, debug=debug)

        # Aggregate the feature dictionaries into a single dictionary
        feature_dictionary = dict([item for sublist in [feature_dict.items() for feature_dict in [auc_features, time_features, beat_features, prominence_feature, slope_features, datum_features, width_features, sqi_features]] for item in sublist])
        
        return feature_dictionary
