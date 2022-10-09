from amplitudes_widths_prominences import *
from pulse_processing import *
from upslopes_downslopes_rise_times_auc import *
from data_methods import *
import math

def run_protocol(window_size, iicp, distal, proximal, subtracted):
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

        chunk_start = 0
        chunk_end = window_size

        print("######################  Patient: " + patient + "  ########################")
        for window in range(num_windows): 
            print(str(window+1) + "/" + str(num_windows))

            iicp_chunk = iicp_data[chunk_start:chunk_end]
            distal_chunk = distal_data[chunk_start:chunk_end]
            proximal_chunk = proximal_data[chunk_start:chunk_end]
            subtracted_chunk = subtracted_data[chunk_start:chunk_end]

            if len(iicp_chunk) != 0:
                print("Chunk start: " + str(chunk_start) + " Chunk end: " + str(chunk_end))

                distal_chunk = band_pass_filter(distal_chunk, 2, 100, 0.5, 12)
                distal_chunk = (distal_chunk - distal_chunk.min())/(distal_chunk.max() - distal_chunk.min())

                proximal_chunk = band_pass_filter(proximal_chunk, 2, 100, 0.5, 12)
                proximal_chunk = (proximal_chunk - proximal_chunk.min())/(proximal_chunk.max() - proximal_chunk.min())

                subtracted_chunk = band_pass_filter(subtracted_chunk, 2, 100, 0.5, 12)
                subtracted_chunk = (subtracted_chunk - subtracted_chunk.min())/(subtracted_chunk.max() - subtracted_chunk.min())

                distal_amplitude, distal_half_width = get_amplitudes_widths_prominences(distal_chunk,fs=100,visualise=0)
                proximal_amplitude, proximal_half_width = get_amplitudes_widths_prominences(proximal_chunk,fs=100,visualise=0)
                subtracted_amplitude, subtracted_half_width = get_amplitudes_widths_prominences(subtracted_chunk,fs=100,visualise=0)

                distal_upslope, distal_downslope, distal_rise_time, distal_decay_time, distal_auc, distal_sys_auc, distal_dia_auc, distal_auc_ratio, distal_second_derivative_ratio = get_upslopes_downslopes_rise_times_auc(distal_chunk,fs=100,visualise=0)
                proximal_upslope, proximal_downslope, proximal_rise_time, proximal_decay_time, proximal_auc, proximal_sys_auc, proximal_dia_auc, proximal_auc_ratio, proximal_second_derivative_ratio = get_upslopes_downslopes_rise_times_auc(proximal_chunk,fs=100,visualise=0)
                subtracted_upslope, subtracted_downslope, subtracted_rise_time, subtracted_decay_time, subtracted_auc, subtracted_sys_auc, subtracted_dia_auc, subtracted_auc_ratio, subtracted_second_derivative_ratio = get_upslopes_downslopes_rise_times_auc(subtracted_chunk,fs=100,visualise=0)

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

        features_df_distal.to_csv("Features/Distal/Patient_"+patient+"_Features_Distal.csv")
        features_df_proximal.to_csv("Features/Proximal/Patient_"+patient+"_Features_Proximal.csv")
        features_df_subtracted.to_csv("Features/Subtracted/Patient_"+patient+"_Features_Subtracted.csv")

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

    features_df_distal_all.to_csv("Features/Joint_Features/ALL_Patients_Features_Distal.csv")
    features_df_proximal_all.to_csv("Features/Joint_Features/ALL_Patients_Features_Proximal.csv")
    features_df_subtracted_all.to_csv("Features/Joint_Features/ALL_Patients_Features_Subtracted.csv")
