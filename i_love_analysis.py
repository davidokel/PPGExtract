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

distal_features = load_csv("Features/Joint_Features/WIDTHS__Updated_extraction_V4_CLEANED_DISTAL.csv")
proximal_features = load_csv("Features/Joint_Features/WIDTHS__Updated_extraction_V4_CLEANED_PROXIMAL.csv")
subtracted_features = load_csv("Features/Joint_Features/WIDTHS__Updated_extraction_V4_CLEANED_SUBTRACTED.csv")

#distal_features = load_csv("Features/Joint_Features/ALL_Patients_Features_Distal.csv").dropna()
#proximal_features = load_csv("Features/Joint_Features/ALL_Patients_Features_Proximal.csv").dropna()
#subtracted_features = load_csv("Features/Joint_Features/ALL_Patients_Features_Subtracted.csv").dropna()

# Collecting all indexes of rows in dataframes which have a nan value
nan_indexes = []
for i in distal_features.index:
    # Check if dataframe row includes a nan value
    if distal_features.loc[i].isnull().values.any():
        nan_indexes.append(i)
for i in proximal_features.index:
    # Check if dataframe row includes a nan value
    if proximal_features.loc[i].isnull().values.any():
        nan_indexes.append(i)
for i in subtracted_features.index:
    # Check if dataframe row includes a nan value
    if subtracted_features.loc[i].isnull().values.any():
        nan_indexes.append(i)

nan_set = set(nan_indexes)

# Remove all rows with nan values from all dataframes
distal_features = distal_features.drop(nan_set)
proximal_features = proximal_features.drop(nan_set)
subtracted_features = subtracted_features.drop(nan_set)

# GROUP FEATURE DATA BY ICP VALUE > 20 AND < 20
distal_below_20 = distal_features.loc[distal_features['IICP Data'] <= 20]
distal_above_20 = distal_features.loc[distal_features['IICP Data'] >= 20]

proximal_below_20 = proximal_features.loc[proximal_features['IICP Data'] <= 20]
proximal_above_20 = proximal_features.loc[proximal_features['IICP Data'] >= 20]

subtracted_below_20 = subtracted_features.loc[subtracted_features['IICP Data'] <= 20]
subtracted_above_20 = subtracted_features.loc[subtracted_features['IICP Data'] >= 20]

features = distal_below_20.columns[0:len(distal_below_20.columns)-1]

for column in range(len(features)):
    boxplot_data = [proximal_below_20[features[column]].tolist(), proximal_above_20[features[column]].tolist(), subtracted_below_20[features[column]].tolist(), subtracted_above_20[features[column]].tolist(), distal_below_20[features[column]].tolist(), distal_above_20[features[column]].tolist()]
    
    #fig = plt.figure()
    plt.subplot(3,4,column+1)
    plt.boxplot(boxplot_data, showfliers=False)
    plt.title("Boxplots of " + features[column] + " for ICP < 20 and ICP > 20", fontsize=10)
    plt.xticks([1, 2, 3, 4, 5, 6], ["Proximal ICP < 20", "Proximal ICP > 20", "Subtracted ICP < 20", "Subtracted > 20", "Distal < 20", "Distal > 20"], rotation=45, fontsize=6)

plt.subplots_adjust(wspace=0.3, hspace=0.6)
plt.suptitle("Boxplots of features for ICP < 20 and ICP > 20 (Proximal, Distal and Subtracted data)", fontsize=15)
manager = plt.get_current_fig_manager()
manager.window.showMaximized()
plt.show()

# Mann-Whitney test for each feature (above and below 20 ICP) for each dataset
distal_test_statistics = []
distal_p_values = []

proximal_test_statistics = []
proximal_p_values = []

subtracted_test_statistics = []
subtracted_p_values = []

for column in range(len(features)):
    below_data = distal_below_20[features[column]]
    above_data = distal_above_20[features[column]]
    distal_statistic, distal_pvalue = stats.mannwhitneyu(below_data.tolist(), above_data.tolist())
    boxplot_data = [below_data.tolist(), above_data.tolist()]

    plt.subplot(3,4,column+1)
    plt.boxplot(boxplot_data, showfliers=False)
    plt.title("Boxplots of " + features[column] + " (P-value: " + str(distal_pvalue) + " )", fontsize=8)
    plt.xticks([1, 2], ["ICP < 20", "ICP > 20"])

    distal_test_statistics.append(float(distal_statistic))
    distal_p_values.append(float(distal_pvalue))

plt.subplots_adjust(wspace=0.3, hspace=0.4)
plt.suptitle("Mann-Whitney U-Test, P-Values for ICP < 20 and ICP > 20 (DISTAL)", fontsize=15)
manager = plt.get_current_fig_manager()
manager.window.showMaximized()
plt.show()

for column in range(len(features)):
    below_data = proximal_below_20[features[column]]
    above_data = proximal_above_20[features[column]]
    proximal_statistic, proximal_pvalue = stats.mannwhitneyu(below_data.tolist(), above_data.tolist())
    boxplot_data = [below_data.tolist(), above_data.tolist()]

    plt.subplot(3,4,column+1)
    plt.boxplot(boxplot_data, showfliers=False)
    plt.title("Boxplots of " + features[column] + " (P-value: " + str(proximal_pvalue) + " )", fontsize=8)
    plt.xticks([1, 2], ["ICP < 20", "ICP > 20"])

    proximal_test_statistics.append(float(proximal_statistic))
    proximal_p_values.append(float(proximal_pvalue))

plt.subplots_adjust(wspace=0.3, hspace=0.4)
plt.suptitle("Mann-Whitney U-Test, P-Values for ICP < 20 and ICP > 20 (PROXIMAL)", fontsize=15)
manager = plt.get_current_fig_manager()
manager.window.showMaximized()
plt.show()

for column in range(len(features)):
    below_data = subtracted_below_20[features[column]]
    above_data = subtracted_above_20[features[column]]
    subtracted_statistic, subtracted_pvalue = stats.mannwhitneyu(below_data.tolist(), above_data.tolist())
    boxplot_data = [below_data.tolist(), above_data.tolist()]

    plt.subplot(3,4,column+1)
    plt.boxplot(boxplot_data, showfliers=False)
    plt.title("Boxplots of " + features[column] + " (P-value: " + str(subtracted_pvalue) + " )", fontsize=8)
    plt.xticks([1, 2], ["ICP < 20", "ICP > 20"])

    subtracted_test_statistics.append(float(subtracted_statistic))
    subtracted_p_values.append(float(subtracted_pvalue))

plt.subplots_adjust(wspace=0.3, hspace=0.4)
plt.suptitle("Mann-Whitney U-Test, P-Values for ICP < 20 and ICP > 20 (SUBTRACTED)", fontsize=15)
manager = plt.get_current_fig_manager()
manager.window.showMaximized()
plt.show()

mann_whitney_results_distal = {'FEATURE':features, 'TEST-STATISTIC':distal_test_statistics, 'P_VALUE':distal_p_values}
mann_whitney_results_df_distal = pd.DataFrame(mann_whitney_results_distal)
mann_whitney_results_df_distal.to_csv("Analysis/Mann_Whitney_U_test/Distal/Mann_Whitney_U_test_RESULTS_DISTAL.csv")

mann_whitney_results_proximal = {'FEATURE':features, 'TEST-STATISTIC':proximal_test_statistics, 'P_VALUE':proximal_p_values}
mann_whitney_results_df_proximal = pd.DataFrame(mann_whitney_results_proximal)
mann_whitney_results_df_proximal.to_csv("Analysis/Mann_Whitney_U_test/Proximal/Mann_Whitney_U_test_RESULTS_PROXIMAL.csv")

mann_whitney_results_subtracted = {'FEATURE':features, 'TEST-STATISTIC':subtracted_test_statistics, 'P_VALUE':subtracted_p_values}
mann_whitney_results_df_subtracted = pd.DataFrame(mann_whitney_results_subtracted)
mann_whitney_results_df_subtracted.to_csv("Analysis/Mann_Whitney_U_test/Subtracted/Mann_Whitney_U_test_RESULTS_SUBTRACTED.csv")

# KRUSKAL-WALLIS TEST

below_20_test_statistics = []
below_20_p_values = []

# NORMAL_ICP LEVELS INTER POSITION
for column in range(len(features)):

    proximal_below_data = proximal_below_20[features[column]].tolist()
    subtracted_below_data = subtracted_below_20[features[column]].tolist()
    distal_below_data = distal_below_20[features[column]].tolist()

    below_statistic, below_pvalue = stats.kruskal(proximal_below_data, subtracted_below_data, distal_below_data)

    boxplot_data = [proximal_below_data, subtracted_below_data, distal_below_data]

    plt.subplot(3,4,column+1)
    plt.boxplot(boxplot_data, showfliers=False)
    plt.title("Boxplots of " + features[column] + " (P-value: " + str(below_pvalue) + " )", fontsize=8)
    plt.xticks([1, 2, 3], ["Proximal ICP < 20", "Subtracted ICP < 20", "Distal < 20"], rotation=45, fontsize=6)

    below_20_test_statistics.append(float(below_statistic))
    below_20_p_values.append(float(below_pvalue))

plt.subplots_adjust(wspace=0.3, hspace=0.6)
plt.suptitle("Kruskal-Wallis, P-Values for ICP < 20 (Proximal, Subtracted and Distal)", fontsize=15)
manager = plt.get_current_fig_manager()
manager.window.showMaximized()
plt.show()

kruskal_below_results = {'FEATURE':features, 'TEST-STATISTIC':below_20_test_statistics, 'P_VALUE':below_20_p_values}
kruskal_below_results_df = pd.DataFrame(kruskal_below_results)
kruskal_below_results_df.to_csv("Analysis/Kruskal_Wallis/Kruskall_RESULTS_BELOW_20.csv")


above_20_test_statistics = []
above_20_p_values = []

# HYPER_ICP LEVELS INTER POSITION
for column in range(len(features)):

    proximal_above_data = proximal_above_20[features[column]].tolist()
    subtracted_above_data = subtracted_above_20[features[column]].tolist()
    distal_above_data = distal_above_20[features[column]].tolist()

    above_statistic, above_pvalue = stats.kruskal(proximal_above_data, subtracted_above_data, distal_above_data)

    boxplot_data = [proximal_above_data, subtracted_above_data, distal_above_data]

    plt.subplot(3,4,column+1)
    plt.boxplot(boxplot_data, showfliers=False)
    plt.title("Boxplots of " + features[column] + " (P-value: " + str(above_pvalue) + " )", fontsize=8)
    plt.xticks([1, 2, 3], ["Proximal ICP > 20", "Subtracted ICP > 20", "Distal > 20"], rotation=45, fontsize=6)

    above_20_test_statistics.append(float(above_statistic))
    above_20_p_values.append(float(above_pvalue))

plt.subplots_adjust(wspace=0.3, hspace=0.6)
plt.suptitle("Kruskal-Wallis, P-Values for ICP > 20 (Proximal, Subtracted and Distal)", fontsize=15)
manager = plt.get_current_fig_manager()
manager.window.showMaximized()
plt.show()

kruskal_above_results = {'FEATURE':features, 'TEST-STATISTIC':below_20_test_statistics, 'P_VALUE':below_20_p_values}
kruskal_above_results_df = pd.DataFrame(kruskal_above_results)
kruskal_above_results_df.to_csv("Analysis/Kruskal_Wallis/Kruskall_RESULTS_ABOVE_20.csv")













"""# U mann whitney test for (FEATURE AND ICP) for each dataset
distal_test_statistics = []
distal_p_values = []

proximal_test_statistics = []
proximal_p_values = []

subtracted_test_statistics = []
subtracted_p_values = []



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
        distal_features_boxcox[column] = stats.boxcox(distal_features[column].tolist())[0]"""

