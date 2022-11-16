from amplitudes_widths_prominences import *
from pulse_processing import *
from upslopes_downslopes_rise_times_auc import *
from data_methods import *
import math
from sqis import get_sqis

def run_protocol(window_size, iicp, distal, proximal, subtracted, save_name):
    columns = iicp.columns

    features_list = ['Amplitude', 'Half-peak width', 'Upslope', 'Downslope', 'Rise time', 'Decay time', 'AUC', 'Sys AUC', 'Dia AUC', 'AUC Ratio', 'Second Derivative Ratio', 'IICP Data']
    features_df_distal_all = pd.DataFrame(columns=features_list)
    features_df_proximal_all = pd.DataFrame(columns=features_list)
    features_df_subtracted_all = pd.DataFrame(columns=features_list)

    all_amplitudes_dis,all_amplitudes_pro,all_amplitudes_sub = [],[],[]
    all_half_widths_dis, all_half_widths_pro, all_half_widths_sub = [],[],[]
    all_upslopes_dis, all_upslopes_pro, all_upslopes_sub = [],[],[]
    all_downslopes_dis, all_downslopes_pro, all_downslopes_sub = [],[],[]
    all_rise_times_dis, all_rise_times_pro, all_rise_times_sub = [],[],[]
    all_decay_times_dis, all_decay_times_pro, all_decay_times_sub = [],[],[]
    all_aucs_dis, all_aucs_pro, all_aucs_sub = [],[],[]
    all_sys_aucs_dis, all_sys_aucs_pro, all_sys_aucs_sub = [],[],[]
    all_dia_aucs_dis, all_dia_aucs_pro, all_dia_aucs_sub = [],[],[]
    all_auc_ratios_dis, all_auc_ratios_pro, all_auc_ratios_sub = [],[],[]
    all_second_derivative_ratios_dis, all_second_derivative_ratios_pro, all_second_derivative_ratios_sub = [],[],[]
    all_iicp_data = []
    all_distal_skews, all_distal_kurts, all_distal_snrs, all_distal_zcrs, all_distal_ents, all_distal_pis = [],[],[],[],[],[]
    all_proximal_skews, all_proximal_kurts, all_proximal_snrs, all_proximal_zcrs, all_proximal_ents, all_proximal_pis = [],[],[],[],[],[]
    all_subtracted_skews, all_subtracted_kurts, all_subtracted_snrs, all_subtracted_zcrs, all_subtracted_ents, all_subtracted_pis = [],[],[],[],[],[]

    for column in range(len(columns)):
        features_df_distal = pd.DataFrame(columns=features_list)
        features_df_proximal = pd.DataFrame(columns=features_list)
        features_df_subtracted = pd.DataFrame(columns=features_list)

        iicp_data = iicp[iicp.columns[column]].dropna().to_numpy()
        distal_data = distal[distal.columns[column]].dropna().to_numpy()
        proximal_data = proximal[proximal.columns[column]].dropna().to_numpy()
        subtracted_data = subtracted[subtracted.columns[column]].dropna().to_numpy()

        num_windows = int(math.floor(len(iicp_data)/window_size))

        patient = columns[column]

        amplitudes_dis, amplitudes_pro, amplitudes_sub = [],[],[]
        half_widths_dis, half_widths_pro, half_widths_sub = [],[],[]
        upslopes_dis, upslopes_pro, upslopes_sub = [],[],[]
        downslopes_dis, downslopes_pro, downslopes_sub = [],[],[]
        rise_times_dis, rise_times_pro, rise_times_sub = [],[],[]
        decay_times_dis, decay_times_pro, decay_times_sub = [],[],[]
        aucs_dis, aucs_pro, aucs_sub = [],[],[]
        sys_aucs_dis, sys_aucs_pro, sys_aucs_sub = [],[],[]
        dia_aucs_dis, dia_aucs_pro, dia_aucs_sub = [],[],[]
        auc_ratios_dis, auc_ratios_pro, auc_ratios_sub = [],[],[]
        second_derivative_ratios_dis, second_derivative_ratios_pro, second_derivative_ratios_sub = [],[],[]
        iicp_value = []
        distal_skews, distal_kurts, distal_snrs, distal_zcrs, distal_ents, distal_pis = [],[],[],[],[],[]
        proximal_skews, proximal_kurts, proximal_snrs, proximal_zcrs, proximal_ents, proximal_pis = [],[],[],[],[],[]
        subtracted_skews, subtracted_kurts, subtracted_snrs, subtracted_zcrs, subtracted_ents, subtracted_pis = [],[],[],[],[],[]

        chunk_start = 0
        chunk_end = window_size

        print("######################  Patient: " + patient + "  ########################")
        for window in range(num_windows): 
            #print(str(window+1) + "/" + str(num_windows))

            iicp_chunk = iicp_data[chunk_start:chunk_end]
            distal_chunk = distal_data[chunk_start:chunk_end]
            proximal_chunk = proximal_data[chunk_start:chunk_end]
            subtracted_chunk = subtracted_data[chunk_start:chunk_end]

            if len(iicp_chunk) != 0:
                #print("Chunk start: " + str(chunk_start) + " Chunk end: " + str(chunk_end))

                distal_chunk_filt = band_pass_filter(distal_chunk, 2, 100, 0.5, 12)
                distal_chunk_norm = normalise_data(distal_chunk_filt, fs=100)
                #distal_chunk = band_pass_filter(distal_chunk, 2, 100, 0.5, 12)
                #distal_chunk = (distal_chunk - distal_chunk.min())/(distal_chunk.max() - distal_chunk.min())

                proximal_chunk_filt = band_pass_filter(proximal_chunk, 2, 100, 0.5, 12)
                proximal_chunk_norm = normalise_data(proximal_chunk_filt, fs=100)
                #proximal_chunk = band_pass_filter(proximal_chunk, 2, 100, 0.5, 12)
                #proximal_chunk = (proximal_chunk - proximal_chunk.min())/(proximal_chunk.max() - proximal_chunk.min())

                subtracted_chunk_filt = band_pass_filter(subtracted_chunk, 2, 100, 0.5, 12)
                subtracted_chunk_norm = normalise_data(subtracted_chunk_filt, fs=100)
                #subtracted_chunk = band_pass_filter(subtracted_chunk, 2, 100, 0.5, 12)
                #subtracted_chunk = (subtracted_chunk - subtracted_chunk.min())/(subtracted_chunk.max() - subtracted_chunk.min())

                distal_amplitude, distal_half_width = get_amplitudes_widths_prominences(distal_chunk_norm,fs=100,visualise=0)
                proximal_amplitude, proximal_half_width = get_amplitudes_widths_prominences(proximal_chunk_norm,fs=100,visualise=0)
                subtracted_amplitude, subtracted_half_width = get_amplitudes_widths_prominences(subtracted_chunk_norm,fs=100,visualise=0)

                distal_upslope, distal_downslope, distal_rise_time, distal_decay_time, distal_auc, distal_sys_auc, distal_dia_auc, distal_auc_ratio, distal_second_derivative_ratio = get_upslopes_downslopes_rise_times_auc(distal_chunk_norm,fs=100,visualise=0)
                proximal_upslope, proximal_downslope, proximal_rise_time, proximal_decay_time, proximal_auc, proximal_sys_auc, proximal_dia_auc, proximal_auc_ratio, proximal_second_derivative_ratio = get_upslopes_downslopes_rise_times_auc(proximal_chunk_norm,fs=100,visualise=0)
                subtracted_upslope, subtracted_downslope, subtracted_rise_time, subtracted_decay_time, subtracted_auc, subtracted_sys_auc, subtracted_dia_auc, subtracted_auc_ratio, subtracted_second_derivative_ratio = get_upslopes_downslopes_rise_times_auc(subtracted_chunk_norm,fs=100,visualise=0)
                
                #skew, kurt, snr, zcr, ent, pi, sqi_dictionary
                distal_skew, distal_kurt, distal_snr, distal_zcr, distal_ent, distal_pi,_ = get_sqis(distal_chunk,fs=100,visualise=0)
                proximal_skew, proximal_kurt, proximal_snr, proximal_zcr, proximal_ent, proximal_pi,_ = get_sqis(proximal_chunk,fs=100,visualise=0)
                subtracted_skew, subtracted_kurt, subtracted_snr, subtracted_zcr, subtracted_ent, subtracted_pi,_ = get_sqis(subtracted_chunk,fs=100,visualise=0)

                amplitudes_dis.append(distal_amplitude)
                amplitudes_pro.append(proximal_amplitude)
                amplitudes_sub.append(subtracted_amplitude)

                half_widths_dis.append(distal_half_width)
                half_widths_pro.append(proximal_half_width)
                half_widths_sub.append(subtracted_half_width)

                upslopes_dis.append(distal_upslope)
                upslopes_pro.append(proximal_upslope)
                upslopes_sub.append(subtracted_upslope)

                downslopes_dis.append(distal_downslope)
                downslopes_pro.append(proximal_downslope)
                downslopes_sub.append(subtracted_downslope)

                rise_times_dis.append(distal_rise_time)
                rise_times_pro.append(proximal_rise_time)
                rise_times_sub.append(subtracted_rise_time)

                decay_times_dis.append(distal_decay_time)
                decay_times_pro.append(proximal_decay_time)
                decay_times_sub.append(subtracted_decay_time)

                aucs_dis.append(distal_auc)
                aucs_pro.append(proximal_auc)
                aucs_sub.append(subtracted_auc)

                sys_aucs_dis.append(distal_sys_auc)
                sys_aucs_pro.append(proximal_sys_auc)
                sys_aucs_sub.append(subtracted_sys_auc)

                dia_aucs_dis.append(distal_dia_auc)
                dia_aucs_pro.append(proximal_dia_auc)
                dia_aucs_sub.append(subtracted_dia_auc)

                auc_ratios_dis.append(distal_auc_ratio)
                auc_ratios_pro.append(proximal_auc_ratio)
                auc_ratios_sub.append(subtracted_auc_ratio)

                second_derivative_ratios_dis.append(distal_second_derivative_ratio)
                second_derivative_ratios_pro.append(proximal_second_derivative_ratio)
                second_derivative_ratios_sub.append(subtracted_second_derivative_ratio)

                if len(iicp_chunk) > 0:
                    #iicp_value.append(iicp_chunk[len(iicp_chunk)-1])
                    iicp_value.append(np.mean(iicp_chunk))
                else:
                    iicp_value.append(np.NaN)

                distal_skews.append(distal_skew)
                proximal_skews.append(proximal_skew)
                subtracted_skews.append(subtracted_skew)
                
                distal_kurts.append(distal_kurt)
                proximal_kurts.append(proximal_kurt)
                subtracted_kurts.append(subtracted_kurt)

                distal_snrs.append(distal_snr)
                proximal_snrs.append(proximal_snr)
                subtracted_snrs.append(subtracted_snr)

                distal_zcrs.append(distal_zcr)
                proximal_zcrs.append(proximal_zcr)
                subtracted_zcrs.append(subtracted_zcr)

                distal_ents.append(distal_ent)
                proximal_ents.append(proximal_ent)
                subtracted_ents.append(subtracted_ent)

                distal_pis.append(distal_pi)
                proximal_pis.append(proximal_pi)
                subtracted_pis.append(subtracted_pi)
                
            chunk_start += window_size
            chunk_end += window_size

        all_amplitudes_dis.extend(amplitudes_dis)
        all_amplitudes_pro.extend(amplitudes_pro)
        all_amplitudes_sub.extend(amplitudes_sub)

        all_half_widths_dis.extend(half_widths_dis)
        all_half_widths_pro.extend(half_widths_pro)
        all_half_widths_sub.extend(half_widths_sub)

        all_upslopes_dis.extend(upslopes_dis)
        all_upslopes_pro.extend(upslopes_pro)
        all_upslopes_sub.extend(upslopes_sub)

        all_downslopes_dis.extend(downslopes_dis)
        all_downslopes_pro.extend(downslopes_pro)
        all_downslopes_sub.extend(downslopes_sub)

        all_rise_times_dis.extend(rise_times_dis)
        all_rise_times_pro.extend(rise_times_pro)
        all_rise_times_sub.extend(rise_times_sub)

        all_decay_times_dis.extend(decay_times_dis)
        all_decay_times_pro.extend(decay_times_pro)
        all_decay_times_sub.extend(decay_times_sub)

        all_aucs_dis.extend(aucs_dis)
        all_aucs_pro.extend(aucs_pro)
        all_aucs_sub.extend(aucs_sub)

        all_sys_aucs_dis.extend(sys_aucs_dis)
        all_sys_aucs_pro.extend(sys_aucs_pro)
        all_sys_aucs_sub.extend(sys_aucs_sub)

        all_dia_aucs_dis.extend(dia_aucs_dis)
        all_dia_aucs_pro.extend(dia_aucs_pro)
        all_dia_aucs_sub.extend(dia_aucs_sub)

        all_auc_ratios_dis.extend(auc_ratios_dis)
        all_auc_ratios_pro.extend(auc_ratios_pro)   
        all_auc_ratios_sub.extend(auc_ratios_sub)

        all_second_derivative_ratios_dis.extend(second_derivative_ratios_dis)
        all_second_derivative_ratios_pro.extend(second_derivative_ratios_pro)
        all_second_derivative_ratios_sub.extend(second_derivative_ratios_sub)

        all_iicp_data.extend(iicp_value)

        all_distal_skews.extend(distal_skews)
        all_proximal_skews.extend(proximal_skews)
        all_subtracted_skews.extend(subtracted_skews)

        all_distal_kurts.extend(distal_kurts)
        all_proximal_kurts.extend(proximal_kurts)
        all_subtracted_kurts.extend(subtracted_kurts)

        all_distal_snrs.extend(distal_snrs)
        all_proximal_snrs.extend(proximal_snrs)
        all_subtracted_snrs.extend(subtracted_snrs)

        all_distal_zcrs.extend(distal_zcrs)
        all_proximal_zcrs.extend(proximal_zcrs)
        all_subtracted_zcrs.extend(subtracted_zcrs)

        all_distal_ents.extend(distal_ents)
        all_proximal_ents.extend(proximal_ents)
        all_subtracted_ents.extend(subtracted_ents)

        all_distal_pis.extend(distal_pis)
        all_proximal_pis.extend(proximal_pis)
        all_subtracted_pis.extend(subtracted_pis)
        
        features_df_distal['Amplitude'] = amplitudes_dis
        features_df_distal['Half-peak width'] = half_widths_dis
        features_df_distal['Upslope'] = upslopes_dis
        features_df_distal['Downslope'] = downslopes_dis
        features_df_distal['Rise time'] = rise_times_dis
        features_df_distal['Decay time'] = decay_times_dis
        features_df_distal['AUC'] = aucs_dis
        features_df_distal['Sys AUC'] = sys_aucs_dis
        features_df_distal['Dia AUC'] = dia_aucs_dis
        features_df_distal['AUC Ratio'] = auc_ratios_dis
        features_df_distal['Second Derivative Ratio'] = second_derivative_ratios_dis
        features_df_distal['IICP Data'] = iicp_value
        features_df_distal['Skew'] = distal_skews
        features_df_distal['Kurtosis'] = distal_kurts
        features_df_distal['SNR'] = distal_snrs
        features_df_distal['ZCR'] = distal_zcrs
        features_df_distal['Entropy'] = distal_ents
        features_df_distal['PI'] = distal_pis

        features_df_proximal['Amplitude'] = amplitudes_pro
        features_df_proximal['Half-peak width'] = half_widths_pro
        features_df_proximal['Upslope'] = upslopes_pro
        features_df_proximal['Downslope'] = downslopes_pro
        features_df_proximal['Rise time'] = rise_times_pro
        features_df_proximal['Decay time'] = decay_times_pro
        features_df_proximal['AUC'] = aucs_pro
        features_df_proximal['Sys AUC'] = sys_aucs_pro
        features_df_proximal['Dia AUC'] = dia_aucs_pro
        features_df_proximal['AUC Ratio'] = auc_ratios_pro
        features_df_proximal['Second Derivative Ratio'] = second_derivative_ratios_pro
        features_df_proximal['IICP Data'] = iicp_value
        features_df_proximal['Skew'] = proximal_skews
        features_df_proximal['Kurtosis'] = proximal_kurts
        features_df_proximal['SNR'] = proximal_snrs
        features_df_proximal['ZCR'] = proximal_zcrs
        features_df_proximal['Entropy'] = proximal_ents
        features_df_proximal['PI'] = proximal_pis

        features_df_subtracted['Amplitude'] = amplitudes_sub
        features_df_subtracted['Half-peak width'] = half_widths_sub
        features_df_subtracted['Upslope'] = upslopes_sub
        features_df_subtracted['Downslope'] = downslopes_sub
        features_df_subtracted['Rise time'] = rise_times_sub
        features_df_subtracted['Decay time'] = decay_times_sub
        features_df_subtracted['AUC'] = aucs_sub
        features_df_subtracted['Sys AUC'] = sys_aucs_sub
        features_df_subtracted['Dia AUC'] = dia_aucs_sub
        features_df_subtracted['AUC Ratio'] = auc_ratios_sub
        features_df_subtracted['Second Derivative Ratio'] = second_derivative_ratios_sub
        features_df_subtracted['IICP Data'] = iicp_value
        features_df_subtracted['Skew'] = subtracted_skews
        features_df_subtracted['Kurtosis'] = subtracted_kurts
        features_df_subtracted['SNR'] = subtracted_snrs
        features_df_subtracted['ZCR'] = subtracted_zcrs
        features_df_subtracted['Entropy'] = subtracted_ents
        features_df_subtracted['PI'] = subtracted_pis

        features_df_distal.to_csv("Features/Distal/" + save_name + "_" +patient+"_Features_Distal.csv")
        features_df_proximal.to_csv("Features/Proximal/" + save_name + "_" +patient+"_Features_Proximal.csv")
        features_df_subtracted.to_csv("Features/Subtracted/" + save_name + "_" +patient+"_Features_Subtracted.csv")

    features_df_distal_all['Amplitude'] = all_amplitudes_dis
    features_df_distal_all['Half-peak width'] = all_half_widths_dis
    features_df_distal_all['Upslope'] = all_upslopes_dis
    features_df_distal_all['Downslope'] = all_downslopes_dis
    features_df_distal_all['Rise time'] = all_rise_times_dis
    features_df_distal_all['Decay time'] = all_decay_times_dis
    features_df_distal_all['AUC'] = all_aucs_dis
    features_df_distal_all['Sys AUC'] = all_sys_aucs_dis
    features_df_distal_all['Dia AUC'] = all_dia_aucs_dis
    features_df_distal_all['AUC Ratio'] = all_auc_ratios_dis
    features_df_distal_all['Second Derivative Ratio'] = all_second_derivative_ratios_dis
    features_df_distal_all['IICP Data'] = all_iicp_data
    features_df_distal_all['Skew'] = all_distal_skews
    features_df_distal_all['Kurtosis'] = all_distal_kurts
    features_df_distal_all['SNR'] = all_distal_snrs
    features_df_distal_all['ZCR'] = all_distal_zcrs
    features_df_distal_all['Entropy'] = all_distal_ents
    features_df_distal_all['PI'] = all_distal_pis

    features_df_proximal_all['Amplitude'] = all_amplitudes_pro
    features_df_proximal_all['Half-peak width'] = all_half_widths_pro
    features_df_proximal_all['Upslope'] = all_upslopes_pro
    features_df_proximal_all['Downslope'] = all_downslopes_pro
    features_df_proximal_all['Rise time'] = all_rise_times_pro
    features_df_proximal_all['Decay time'] = all_decay_times_pro
    features_df_proximal_all['AUC'] = all_aucs_pro
    features_df_proximal_all['Sys AUC'] = all_sys_aucs_pro
    features_df_proximal_all['Dia AUC'] = all_dia_aucs_pro
    features_df_proximal_all['AUC Ratio'] = all_auc_ratios_pro
    features_df_proximal_all['Second Derivative Ratio'] = all_second_derivative_ratios_pro
    features_df_proximal_all['IICP Data'] = all_iicp_data
    features_df_proximal_all['Skew'] = all_proximal_skews
    features_df_proximal_all['Kurtosis'] = all_proximal_kurts
    features_df_proximal_all['SNR'] = all_proximal_snrs
    features_df_proximal_all['ZCR'] = all_proximal_zcrs
    features_df_proximal_all['Entropy'] = all_proximal_ents
    features_df_proximal_all['PI'] = all_proximal_pis

    features_df_subtracted_all['Amplitude'] = all_amplitudes_sub
    features_df_subtracted_all['Half-peak width'] = all_half_widths_sub
    features_df_subtracted_all['Upslope'] = all_upslopes_sub
    features_df_subtracted_all['Downslope'] = all_downslopes_sub
    features_df_subtracted_all['Rise time'] = all_rise_times_sub
    features_df_subtracted_all['Decay time'] = all_decay_times_sub
    features_df_subtracted_all['AUC'] = all_aucs_sub
    features_df_subtracted_all['Sys AUC'] = all_sys_aucs_sub
    features_df_subtracted_all['Dia AUC'] = all_dia_aucs_sub
    features_df_subtracted_all['AUC Ratio'] = all_auc_ratios_sub
    features_df_subtracted_all['Second Derivative Ratio'] = all_second_derivative_ratios_sub
    features_df_subtracted_all['IICP Data'] = all_iicp_data
    features_df_subtracted_all['Skew'] = all_subtracted_skews
    features_df_subtracted_all['Kurtosis'] = all_subtracted_kurts
    features_df_subtracted_all['SNR'] = all_subtracted_snrs
    features_df_subtracted_all['ZCR'] = all_subtracted_zcrs
    features_df_subtracted_all['Entropy'] = all_subtracted_ents
    features_df_subtracted_all['PI'] = all_subtracted_pis

    features_df_distal_all.to_csv("Features/Joint_Features/" + save_name + "_DISTAL_NORM.csv")
    features_df_proximal_all.to_csv("Features/Joint_Features/" + save_name + "_PROXIMAL_NORM.csv")
    features_df_subtracted_all.to_csv("Features/Joint_Features/" + save_name + "_SUBTRACTED_NORM.csv")


