from signal_quality_classifiers.dictionary_formatter import dict_to_df
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

def get_pulse_predictions(dict, model_path):

    # Convert dict to df
    df = dict_to_df(dict)
    # Remove the class column from the dataframe
    df = df.drop(['Peak', 'Pre_peak', 'Post_peak', 'Relative_peak', 'raw_pulse_data', 'norm_pulse_data', 'class'], axis=1)
    print(df.columns)

    """['norm_mean', 'norm_median', 'norm_variance', 'raw_mean', 'raw_median',
       'raw_variance', 'secder_norm_mean', 'secder_norm_median',
       'secder_norm_variance', 'skew', 'kurt', 'snr', 'zcr', 'ent', 'pi',
       'auc', 'sys_auc', 'dia_auc', 'auc_ratio', 'half_width', 'rise_time',
       'decay_time', 'prominence', 'sec_der_ratio', 'upslope_length',
       'downslope_length', 'upslope', 'downslope', 'onset_end_slope',
       'upslope_downslope_ratio', 'prepostslope_downslope_ratio',
       'upslope_prepostslope_ratio', 'pulse_length_height_ratio',
       'upslope_downslope_length_ratio', 'upslope_pulse_length_ratio',
       'pulse_downslope_length_ratio', 'height_upslope_ratio',
       'height_downslope_ratio']"""
    
    """['auc', 'auc_ratio', 'decay_time', 'dia_auc', 'downslope',
       'downslope_length', 'ent', 'half_width', 'height_downslope_ratio',
       'height_upslope_ratio', 'kurt', 'onset_end_slope', 'peak_height', 'pi',
       'prepostslope_downslope_ratio', 'prominence',
       'pulse_downslope_length_ratio', 'pulse_length_height_ratio',
       'rise_time', 'sec_der_ratoio', 'skew', 'snr', 'sys_auc', 'upslope',
       'upslope_downslope_length_ratio', 'upslope_downslope_ratio',
       'upslope_length', 'upslope_prepostslope_ratio',
       'upslope_pulse_length_ratio', 'zcr']"""
    
    scaler = StandardScaler()
    df = scaler.fit_transform(df)
    
    classifier_pkl = open(model_path, 'rb')
    classifier = pickle.load(classifier_pkl)

    # Make predictions on the testing data
    y_pred = classifier.predict(df)

    print(y_pred)

    return dict
