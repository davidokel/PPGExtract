import numpy as np
import matplotlib.pyplot as plt
import support_code.data_methods as data_methods
from scipy.integrate import trapz
    
def get_aucs(pulse_data, visualise=0):
    data = pulse_data["raw_pulse_data"]
    peak = pulse_data["Relative_peak"]
    pre = 0
    post = len(pulse_data["norm_pulse_data"])

    data_scaled = data_methods.data_scaler(np.array(data)) # Scale the data to be between 0 and 1 (Used for plotting)

    auc = []
    sys_auc = []
    dia_auc = []
    auc_ratios = []

    if peak:            
        x = range(pre,post)
        y = []
        for index in x:
            y.append(abs(data[index]))
        auc.append(trapz(y,x))

        x = range(pre,peak)
        y = []
        for index in x:
            y.append(abs(data[index]))
        sys_auc.append(trapz(y,x))

        x = range(peak,post)
        y = []
        for index in x:
            y.append(abs(data[index]))
        dia_auc.append(trapz(y,x))

        for area in range(len(dia_auc)):
            auc_ratio = sys_auc[area]/dia_auc[area]
            auc_ratios.append(auc_ratio)

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
            plt.title("Systolic under the curve (S-AUC)")
            plt.plot(data_scaled)
            x = range(pre,peak)
            y = []
            for index in x:
                y.append(abs(data_scaled[index]))
            plt.fill_between(x,y)
            #plt.axis('off')

            plt.subplot(3,1,3)
            plt.title("Diastolic under the curve (D-AUC)")
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

        return float(np.nanmedian(auc)), float(np.nanmedian(sys_auc)), float(np.nanmedian(dia_auc)), float(np.nanmedian(auc_ratios))
    else:
        return np.NaN, np.NaN, np.NaN, np.NaN
