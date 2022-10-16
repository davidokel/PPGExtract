from cmath import nan
from data_methods import load_csv, band_pass_filter
import matplotlib.pyplot as plt
import random as rd
from amplitudes_widths_prominences import get_amplitudes_widths_prominences
from upslopes_downslopes_rise_times_auc import get_upslopes_downslopes_rise_times_auc
import numpy as np
from pulse_processing import *
from data_methods import *
from protocol import *
import scipy.signal as sp
import scipy.stats as stats

distal_features = load_csv("Features/Joint_Features/ALL_Patients_Features_Distal.csv").dropna()
proximal_features = load_csv("Features/Joint_Features/ALL_Patients_Features_Proximal.csv").dropna()
subtracted_features = load_csv("Features/Joint_Features/ALL_Patients_Features_Subtracted.csv").dropna()

# GROUP FEATURE DATA BY ICP VALUE > 20 AND < 20
distal_below_20 = distal_features.loc[distal_features['IICP Data'] < 20]
distal_above_20 = distal_features.loc[distal_features['IICP Data'] > 20]

proximal_below_20 = proximal_features.loc[proximal_features['IICP Data'] < 20]
proximal_above_20 = proximal_features.loc[proximal_features['IICP Data'] > 20]

subtracted_below_20 = subtracted_features.loc[subtracted_features['IICP Data'] < 20]
subtracted_above_20 = subtracted_features.loc[subtracted_features['IICP Data'] > 20]

features = distal_below_20.columns[0:len(distal_below_20.columns)-1]

for column in features:
    boxplot_data = [distal_below_20[column].tolist(), distal_above_20[column].tolist(), proximal_below_20[column].tolist(), proximal_above_20[column].tolist(), subtracted_below_20[column].tolist(), subtracted_above_20[column].tolist()]
    
    fig = plt.figure()
    plt.boxplot(boxplot_data, showfliers=False)
    plt.title("Boxplots of " + column + " for ICP < 20 and ICP > 20")
    plt.xticks([1, 2, 3, 4, 5, 6], ["P NT", "P HT", "D NT", "D HT", "S NT", "S HT"])
    fig.savefig("Analysis/Boxplots/"+column+".png")
    plt.close()
    """manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    plt.show()"""

# Kruskal-Wallis test for each feature (above and below 20 ICP) for each dataset
distal_test_statistics = []
distal_p_values = []

proximal_test_statistics = []
proximal_p_values = []

subtracted_test_statistics = []
subtracted_p_values = []

for column in features:
    distal_statistic, distal_pvalue = stats.kruskal(distal_above_20[column].tolist(), distal_below_20[column].tolist())
    proximal_statistic, proximal_pvalue = stats.kruskal(proximal_above_20[column].tolist(), proximal_below_20[column].tolist())
    subtracted_statistic, subtracted_pvalue = stats.kruskal(subtracted_above_20[column].tolist(), subtracted_below_20[column].tolist())

    fig = plt.figure()
    plt.boxplot([distal_below_20[column].tolist(), distal_above_20[column].tolist()],showfliers=False)
    plt.title("Boxplots of " + column + " (P-value: " + str(distal_pvalue) + " )")
    plt.xticks([1, 2], ["ICP < 20", "ICP > 20"])
    fig.savefig("Analysis/Kruskal_Wallis/Distal/"+column+".png")
    plt.close()

    fig = plt.figure()
    plt.boxplot([proximal_below_20[column].tolist(), proximal_above_20[column].tolist()], showfliers=False)
    plt.title("Boxplots of " + column + " (P-value: " + str(proximal_pvalue) + " )")
    plt.xticks([1, 2], ["ICP < 20", "ICP > 20"])
    fig.savefig("Analysis/Kruskal_Wallis/Proximal/"+column+".png")
    plt.close()

    fig = plt.figure()
    plt.boxplot([subtracted_below_20[column].tolist(), subtracted_above_20[column].tolist()], showfliers=False)
    plt.title("Boxplots of " + column + " (P-value: " + str(subtracted_pvalue) + " )")
    plt.xticks([1, 2], ["ICP < 20", "ICP > 20"])
    fig.savefig("Analysis/Kruskal_Wallis/Subtracted/"+column+".png")
    plt.close()

    distal_test_statistics.append(float(distal_statistic))
    distal_p_values.append(float(distal_pvalue))

    proximal_test_statistics.append(float(proximal_statistic))
    proximal_p_values.append(float(proximal_pvalue))

    subtracted_test_statistics.append(float(subtracted_statistic))
    subtracted_p_values.append(float(subtracted_pvalue))

kruskal_wallis_results_distal = {'FEATURE':features, 'TEST-STATISTIC':distal_test_statistics, 'P_VALUE':distal_p_values}
kruskal_wallis_results_df_distal = pd.DataFrame(kruskal_wallis_results_distal)
kruskal_wallis_results_df_distal.to_csv("Analysis/Kruskal_Wallis/Distal/Kruskal_Wallis_RESULTS_DISTAL.csv")

kruskal_wallis_results_proximal = {'FEATURE':features, 'TEST-STATISTIC':proximal_test_statistics, 'P_VALUE':proximal_p_values}
kruskal_wallis_results_df_proximal = pd.DataFrame(kruskal_wallis_results_proximal)
kruskal_wallis_results_df_proximal.to_csv("Analysis/Kruskal_Wallis/Proximal/Kruskal_Wallis_RESULTS_PROXIMAL.csv")

kruskal_wallis_results_subtracted = {'FEATURE':features, 'TEST-STATISTIC':subtracted_test_statistics, 'P_VALUE':subtracted_p_values}
kruskal_wallis_results_df_subtracted = pd.DataFrame(kruskal_wallis_results_subtracted)
kruskal_wallis_results_df_subtracted.to_csv("Analysis/Kruskal_Wallis/Subtracted/Kruskal_Wallis_RESULTS_SUBTRACTED.csv")

# U mann whitney test for (FEATURE AND ICP) for each dataset
distal_test_statistics = []
distal_p_values = []

proximal_test_statistics = []
proximal_p_values = []

subtracted_test_statistics = []
subtracted_p_values = []

for column in features:
    distal_statistic, distal_pvalue = stats.mannwhitneyu(distal_features[column].tolist(), distal_features['IICP Data'].tolist())
    proximal_statistic, proximal_pvalue = stats.mannwhitneyu(proximal_features[column].tolist(), proximal_features['IICP Data'].tolist())
    subtracted_statistic, subtracted_pvalue = stats.mannwhitneyu(subtracted_features[column].tolist(), subtracted_features['IICP Data'].tolist())

    distal_test_statistics.append(float(distal_statistic))
    distal_p_values.append(float(distal_pvalue))

    proximal_test_statistics.append(float(proximal_statistic))
    proximal_p_values.append(float(proximal_pvalue))

    subtracted_test_statistics.append(float(subtracted_statistic))
    subtracted_p_values.append(float(subtracted_pvalue))

mann_whitney_results_distal = {'FEATURE':features, 'TEST-STATISTIC':distal_test_statistics, 'P_VALUE':distal_p_values}
mann_whitney_results_df_distal = pd.DataFrame(mann_whitney_results_distal)
mann_whitney_results_df_distal.to_csv("Analysis/Mann_Whitney_U_test/Distal/Mann_Whitney_U_test_RESULTS_DISTAL.csv")

mann_whitney_results_proximal = {'FEATURE':features, 'TEST-STATISTIC':proximal_test_statistics, 'P_VALUE':proximal_p_values}
mann_whitney_results_df_proximal = pd.DataFrame(mann_whitney_results_proximal)
mann_whitney_results_df_proximal.to_csv("Analysis/Mann_Whitney_U_test/Proximal/Mann_Whitney_U_test_RESULTS_PROXIMAL.csv")

mann_whitney_results_subtracted = {'FEATURE':features, 'TEST-STATISTIC':subtracted_test_statistics, 'P_VALUE':subtracted_p_values}
mann_whitney_results_df_subtracted = pd.DataFrame(mann_whitney_results_subtracted)
mann_whitney_results_df_subtracted.to_csv("Analysis/Mann_Whitney_U_test/Subtracted/Mann_Whitney_U_test_RESULTS_SUBTRACTED.csv")

# Boxcox transformation of the data
normal_threshold = 0.05

distal_features_boxcox = distal_features.copy()
proximal_features_boxcox = proximal_features.copy()
subtracted_features_boxcox = subtracted_features.copy()

for column in features:
    # SHAPIRO TESTING
    # PLOT DATAFRAME HISTOGRAM
    distal_features.hist(bins=40)
    #plt.show()
    distal_statistic, distal_pvalue = stats.shapiro(distal_features[column].tolist())
    #print("Distal Shapiro Test for " + column + " : " + str(distal_pvalue))
    if distal_pvalue < normal_threshold:
        distal_features_boxcox[column] = stats.boxcox(distal_features[column].tolist())[0]

    proximal_features.hist(bins=40)
    #plt.show()
    proximal_statistic, proximal_pvalue = stats.shapiro(proximal_features[column].tolist())
    #print("Proximal Shapiro Test for " + column + " : " + str(proximal_pvalue))
    if distal_pvalue < normal_threshold:
        distal_features_boxcox[column] = stats.boxcox(distal_features[column].tolist())[0]

    subtracted_features.hist(bins=40)
    #plt.show()
    subtracted_statistic, subtracted_pvalue = stats.shapiro(subtracted_features[column].tolist())
    #print("Subtracted Shapiro Test for " + column + " : " + str(subtracted_pvalue))
    if distal_pvalue < normal_threshold:
        distal_features_boxcox[column] = stats.boxcox(distal_features[column].tolist())[0]


