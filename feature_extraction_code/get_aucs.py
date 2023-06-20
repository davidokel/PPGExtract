# Importing packages
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapz
from support_code.data_methods import data_scaler

def get_aucs(window_pulse_data, visualise=False, debug=False):
    """
    Calculates the area under the curve (AUC) for pulse data within a given window.

    Inputs:
    - window_pulse_data: A dictionary containing pulse data for each pulse within the window.
                         Each key represents a pulse, and the corresponding value is a dictionary
                         containing the pulse's raw data and peak information.
    - visualise: An optional parameter (default is False) to visualize the AUC plots.
    - debug: An optional parameter (default is False) to visualize the AUC plots.

    Outputs:
    - auc_features: A dictionary which includes (if window_pulse_data is not empty)
        - Median AUC: The median of all calculated AUC values
        - Median Systolic AUC: The median of all calculated systolic AUC values
        - Median Diastolic AUC: The median of all calculated diastolic AUC values
        - Median AUC Ratio: The median of the calculated AUC ratios (Systolic AUC / Diastolic AUC)

    Error Handling:
    - If the window_pulse_data is empty, the function returns NaN for all outputs.
    """
    
    ######################################################
    # Initialising lists to store the extracted AUC data #
    ######################################################
    aucs, sys_aucs, dia_aucs, auc_ratios = [], [], [], []

    # Check that window_pulse_data is not empty
    if window_pulse_data:
        # Iterating over the keys of the dictionary, every key represents a pulse within the window
        for key in window_pulse_data:
            #######################
            # Defining pulse data #
            #######################
            pulse_data = window_pulse_data[key]
            data = pulse_data["pulse_data"]
            peak = pulse_data["relative_peak"]
            pre = 0
            post = len(data)

            """Scaling the data by adding a constant to make all values positive
            Scaling does not alter the shape or proportion of the curves: 
            Scaling the PPG data by adding a constant factor does not change the relative shape or proportion of the PPG waveforms. 
            The AUC calculations primarily rely on relative positions, which remain intact regardless of scaling."""
            data_scaled = data_scaler(np.array(data))

            if peak:
                ##############################
                # Calculating the AUC values #
                ##############################
                """Calculating the AUC, systolic AUC (S-AUC), diastolic AUC (D-AUC), and AUC ratios for different segments of the waveform. 
                It iterates over the specified ranges (pre to post and pre to peak) to compute the absolute values of data_scaled
                at each index and then applies the trapezoidal rule (trapz) to calculate the respective AUC values. 
                The AUC ratios are determined by dividing each systolic AUC by its corresponding diastolic AUC."""            
                x = range(pre,post)
                y = []
                for index in x:
                    y.append(abs(data_scaled[index]))
                aucs.append(trapz(y,x))

                x = range(pre,peak)
                y = []
                for index in x:
                    y.append(abs(data_scaled[index]))
                sys_aucs.append(trapz(y,x))

                x = range(peak,post)
                y = []
                for index in x:
                    y.append(abs(data_scaled[index]))
                dia_aucs.append(trapz(y,x))

                for area in range(len(dia_aucs)):
                    auc_ratio = sys_aucs[area]/dia_aucs[area]
                    auc_ratios.append(auc_ratio)

                #########
                # Debug #
                #########
                if debug:
                    print("AUC: " + str(aucs[-1]))
                    print("S-AUC: " + str(sys_aucs[-1]))
                    print("D-AUC: " + str(dia_aucs[-1]))
                    print("AUC Ratio: " + str(auc_ratios[-1]))

                ############
                # Plotting #
                ############
                if visualise == 1:
                    plt.subplot(3,1,1)
                    plt.title("Area under the curve (AUC)")
                    plt.plot(data_scaled)
                    x = range(pre,post)
                    y = []
                    for index in x:
                        y.append(abs(data_scaled[index]))
                    plt.fill_between(x,y)
                    #plt.axis('off')
                    
                    plt.subplot(3,1,2)
                    plt.title("Systolic AUC (S-AUC)")
                    plt.plot(data_scaled)
                    x = range(pre,peak)
                    y = []
                    for index in x:
                        y.append(abs(data_scaled[index]))
                    plt.fill_between(x,y)
                    #plt.axis('off')

                    plt.subplot(3,1,3)
                    plt.title("Diastolic AUC (D-AUC)")
                    plt.plot(data_scaled)
                    x = range(peak,post)
                    y = []
                    for index in x:
                        y.append(abs(data_scaled[index]))
                    plt.fill_between(x,y)

                    plt.subplots_adjust(hspace=0.3)
                    manager = plt.get_current_fig_manager()
                    manager.window.showMaximized()
                    #plt.axis('off')

                    plt.axis('tight')
                    plt.tight_layout()
                    plt.show()
                    
                    # Get the user input to see if they want to move onto the next visualisation or stop visualising
                    user_input = input("Press enter to continue, or type 'stop' to stop visualising: ")
                    if user_input == "stop":
                        visualise = 0
        
        auc_features = {}
        auc_features["AUC"] = float(np.nanmedian(aucs))
        auc_features["S-AUC"] = float(np.nanmedian(sys_aucs))
        auc_features["D-AUC"] = float(np.nanmedian(dia_aucs))
        auc_features["AUC Ratio"] = float(np.nanmedian(auc_ratios))
        return auc_features
    else:
        return {}
