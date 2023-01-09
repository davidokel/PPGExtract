from data_methods import load_csv
import matplotlib.pyplot as plt
import random as rd
import numpy as np
import scipy.signal as sp
import scipy.stats as stats
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import re
import os
import datetime
import statsmodels.api as sm
from ast import literal_eval

def filter_keys(dictionary, dis_prox, prox_only, dis_only):
    # Purpose of function: 
    # Given a dictionary of data and analysis focus (distal and proximal, proximal only or distal only)
    # return the correct keys to run the analysis on.
    #
    # Eg: Given a dictionary with keys (dis_810, dis_850, prox_810 and prox_850) and dis_only = True
    # The expected keys to be returned from the "filter_keys" function will be dis_810 and dis_850.
    #
    # Inputs:
    # dictionary = A dictionary of data with keys in the format dis_+number or prox_+number.
    # dis_prox, prox_only and dis_only = Boolean, only one argument can be True at a time.
    #
    # Outputs:
    # keys_to_use = A list of keys.

    numbers = []
    keys = list(dictionary.keys())

    # Extracting numbers from keys (eg: extracting 810 from dis_810) and adding extracted number to list.
    for key in keys:
        number = re.findall(r'[0-9]+', key)[0]
        numbers.append(number)
    numbers = list(set(numbers))

    keys_to_use = []

    # Iterating over the extracted numbers
    for number in numbers:
        # Defining the two possible key formats dis_+number or prox_+number
        dis_test = 'dis_'+number
        prox_test = 'prox_'+number

        # Dependendant on input argument, checking if suitable key exists in keys if true add to key_to_use list
        if dis_prox == True:
            if dis_test and prox_test in keys:
                keys_to_use.append(dis_test)
                keys_to_use.append(prox_test)
        if prox_only == True:
            if prox_test in keys:
                keys_to_use.append(prox_test)
        if dis_only == True:
            if dis_test in keys:
                keys_to_use.append(dis_test)
    
    if len(keys_to_use) == 1:
        raise Exception("There is only one suitable key. In order to perform analysis > 1 key is needed.")
    elif len(keys_to_use) == 0:
        raise Exception("No suitbale keys were found, ensure that the keys are in the format 'dis_+prox' or 'prox_+dis'.")
    else:
        print("Analysing data associated with keys: ", keys_to_use)
        return keys_to_use

def normality_tests(keys, path_root, dictionary):
    # Purpose of function: 
    # Test for the normality of given data using four methods:
    # 1 -> Plotting of histogram (visual)
    # 2 -> Plotting of QQ-plot (visual)
    # 3 -> Running of Shapiro Wilk test
    # 4 -> Running of Kolmogorov Smirnov test
    #
    # Inputs:
    # keys = A list of keys.
    # path_root = Path to save normality test findings under in String format.
    # dictionary = A dictionary of data with keys in the format dis_+number or prox_+number.
    #
    # Outputs:
    # None

    # Iterating over the extracted numbers
    for key in keys:
        # Isolating the data for the current key
        key_data = dictionary[key]

        # Removing IICP and Patient column from the key_data dataframe as we do not want to calculate the normality of these columns
        normality_test_data = key_data.drop(['IICP', 'Patient'], axis=1)
        columns = normality_test_data.columns

        print("Running normality test for data: ", key)

        # Defining the path to save the normality test results
        path = path_root+"/normality tests/"+str(key)

        # Defining the histogram save path and making the folder if it does not exist
        histogram_path = path+"/histograms/"
        if os.path.exists(histogram_path) == False:
            os.makedirs(histogram_path)
        
        # Defining the QQ-plot save path and making the folder if it does not exist
        qq_path = path+"/qq_plots/"
        if os.path.exists(qq_path) == False:
            os.makedirs(qq_path)

        # Defining dictionaries and lists used to save the results of the Shapiro Wilk test
        shapiro_wilk_results = {}
        shapiro_wilk_result_list = []
        shapiro_wilk_pass_fail_list = []
        # Defining the Shapiro Wilk save path and making the folder if it does not exit
        shapiro_wilk_path = path+"/shapiro_wilk/"
        if os.path.exists(shapiro_wilk_path) == False:
            os.makedirs(shapiro_wilk_path)

        # Defining dictionaries and lists used to save the results of the D'Agostino's K-squared test
        k_squared_results = {}
        k_squared_result_list = []
        k_squared_pass_fail_list = []
        # Defining the Shapiro Wilk save path and making the folder if it does not exit
        k_squared_path = path+"/k_squared/"
        if os.path.exists(k_squared_path) == False:
            os.makedirs(k_squared_path)

        # Defining dictionaries and lists used to save the results of the Kolmogorov Smirnov test
        kolmogorov_smirnov_results = {}
        kolmogorov_smirnov_result_list = []
        kolmogorov_smirnov_pass_fail_list = []
        # Defining the Kolmogorov Smirnov save path and making the folder if it does not exist
        kolmogorov_smirnov_path = path+"/kolmogorov_smirnov/"
        if os.path.exists(kolmogorov_smirnov_path) == False:
            os.makedirs(kolmogorov_smirnov_path)
        
        statistical_test_columns = []

        # Iterating over columns of which normality tests are wanted
        for column in columns:
            # Isolating the data of the given column and dropping nan values (needed for accurate normality tests)
            column_data = normality_test_data[column].dropna()

            # Calculating and saving histogram
            # The data is divided into a pre-specified number of groups called bins.
            # The data is then sorted to each bin and the count of the number of observations in each bin is retained.
            column_data.hist(bins=100)
            title = (str(key)+" "+str(column)+" Histogram")
            plt.xlabel("Value")
            plt.ylabel("Number of Samples")
            plt.title(title)
            plt.savefig(histogram_path+title+".png")
            plt.clf()
            plt.close()

            # Calculating and saving QQ-plot
            # This plot generates its own sample of the idealised distribution that we are comparing with, in this case the Gaussian distribution
            # The idealised samples are divided into groups, called quantiles. Each data point in the sample is paired with a similar member from
            # idealised distribution at the same cumulative distribution. A perfect match for the distribution will be shown by a line of dots on
            # a 45-degree angle from the bottom left of the plot to the top right. 
            sm.qqplot(column_data, line='s')
            title = (str(key)+" "+str(column)+" Q-Q Plot")
            plt.title(title)
            plt.savefig(qq_path+title+".png")
            plt.clf()
            plt.close()

            statistical_test_columns.append(column)

            alpha = 0.05

            # Performing a Shapiro-Wilk test
            # The Shapiro-Wilk (S-W) test evaluates a data sample and quantifies how likely it is that the data was drawn from a Gaussian distribution.
            # The S-W test is believed to be a reliable test for normality, although there is some suggestion that the rest may be suitable for smaller samoles fo data.
            shapiro_wilk_stat, shapiro_wilk_p = stats.shapiro(column_data)
            shapiro_wilk_result_list.append(shapiro_wilk_p)
            if shapiro_wilk_p > alpha:
                shapiro_wilk_pass_fail_list.append("Normally distributed")
            else:
                shapiro_wilk_pass_fail_list.append("Non-normally distributed")

            # Performing D'Agostino's K-squared test
            # The D'Agostino's K-squared test calculates summary statistics from the data, namely kurtosis and skewness, to determine if the data distribution departs from the normal distribution.
            k_squared_stat, k_squared_p = stats.normaltest(column_data)
            k_squared_result_list.append(k_squared_p)
            if k_squared_p > alpha:
                k_squared_pass_fail_list.append("Normally distributed")
            else:
                k_squared_pass_fail_list.append("Non-normally distributed")

            # Performing a Kolmogorov_Smirnov test
            kolmogorov_smirnov_stat, kolmogorov_smirnov_p = stats.kstest(column_data, 'norm')
            kolmogorov_smirnov_result_list.append(kolmogorov_smirnov_p)
            if kolmogorov_smirnov_p > alpha:
                kolmogorov_smirnov_pass_fail_list.append("Normally distributed")
            else:
                kolmogorov_smirnov_pass_fail_list.append("Non-normally distributed")

        # Saving the results of the Shapiro Wilk normality test with columns (Feature, P-value, Distribution)
        shapiro_wilk_results["Feature"] = statistical_test_columns
        shapiro_wilk_results["P-value"] = shapiro_wilk_result_list
        shapiro_wilk_results["Distribution"] = shapiro_wilk_pass_fail_list
        shapiro_df = pd.DataFrame.from_dict(shapiro_wilk_results)
        shapiro_df.to_csv(shapiro_wilk_path+"shapiro_wilk.csv", index=False)

        # Saving the results of the D'Agostino's K-squared test with columns (Feature, P-value, Distribution)
        k_squared_results["Feature"] = statistical_test_columns
        k_squared_results["P-value"] = shapiro_wilk_result_list
        k_squared_results["Distribution"] = shapiro_wilk_pass_fail_list
        k_squared_df = pd.DataFrame.from_dict(k_squared_results)
        k_squared_df.to_csv(k_squared_path+"k_squared.csv", index=False)

        # Saving the results of the Kolmogorov Smirnov normality test with columns (Feature, P-value, Distribution)
        kolmogorov_smirnov_results["Feature"] = statistical_test_columns
        kolmogorov_smirnov_results["P-value"] = kolmogorov_smirnov_result_list
        kolmogorov_smirnov_results["Distribution"] = kolmogorov_smirnov_pass_fail_list
        kolmogorov_df = pd.DataFrame.from_dict(kolmogorov_smirnov_results)
        kolmogorov_df.to_csv(kolmogorov_smirnov_path+"kolmogorov_smirnov.csv", index=False)

def kruskal_wallis(kruskal_path_addition, keys, path_root, dictionary):
    # Purpose of function: 
    # Performing a Kruskal-Wallis test and saving the results as a boxplot and csv
    #
    # Inputs:
    # kruskal_path_addition = Value to append at the end of the results save path, in order to differentiate from previous runs.
    # keys = Dictionary keys
    # path_root = Path to save Kruskal Wallis test findings under in String format.
    # dictionary = A dictionary of data with keys in the format dis_+number or prox_+number.
    #
    # Outputs:
    # None

    columns = list(dictionary[keys[0]].columns)
    # Removing the IICP and Patient columns from the column list as the data of which is not wanted when performing a Kruskal-Wallis test
    columns.remove('IICP')
    columns.remove('Patient')

    # Defining lists used to save the results of the Kruskal-Wallis test
    statistical_test_columns = []
    kruskal_statistic_list = []
    kruskal_pvalue_list = []
    kruskal_pass_fail_list = []

    # Defining the Kruskal-Wallis save path and making the folder if it does not exist
    kruskal_path_root = path_root+"/kruskal wallis test/"
    if os.path.exists(kruskal_path_root) == False:
            os.makedirs(kruskal_path_root)

    i = 0
    for column in columns:
        kruskal_wallis_data = {}
        statistical_test_columns.append(column)
        
        for key in keys:
            key_data = dictionary[key]
            key_data = key_data.drop(['IICP', 'Patient'], axis=1)

            key_column_data = key_data[column].dropna().tolist()
            kruskal_wallis_data[key] = key_column_data
        
        keys = list(kruskal_wallis_data.keys())
        kruskal_calls = []
        for key_num in range(len(keys)):
            kruskal_calls.append("kruskal_wallis_data["+"'"+str(keys[key_num])+"'"+"]")

        alpha = 0.05

        # Once the data for the current columnn from each group has been added to kruskal_wallis_data a Kruskal_Wallis test is performed
        data_call_string = ', '.join([str(elem) for elem in kruskal_calls])
        # In order to make this function dynamic and capable of handling different dictionary sizes, the Kruskal-Wallis test is called using the eval() function
        kruskal_call_string = "stats.kruskal("+data_call_string+")"
        kruskal_wallis_stat, kruskal_wallis_p = eval(kruskal_call_string)
        kruskal_pvalue_list.append(kruskal_wallis_p)
        kruskal_statistic_list.append(kruskal_wallis_stat)

        if kruskal_wallis_p < alpha:
            kruskal_pass_fail_list.append("No significant differences")
        else:
            kruskal_pass_fail_list.append("Significant differences")

        # Creating and saving the boxplot for the Kruskal-Wallis test
        plt.subplot(4,3,i+1)
        boxplot_data = eval("["+data_call_string+"]")
        plt.boxplot(boxplot_data, showfliers=False)
        plt.title("Boxplots of " + column + " (P-value: " + str(np.round(kruskal_wallis_p,4)) + " )", fontsize=8)
        xticks_range = np.arange(1,len(keys)+1)
        plt.xticks(xticks_range, keys, fontsize=6)
        i+=1

    plt.suptitle("Boxplots of features across wavelengths")
    plt.subplots_adjust(hspace=0.5)
    plt.savefig(kruskal_path_root+"kruskal_wallis_boxplots_"+kruskal_path_addition+".png")
    plt.clf()
    
    # Saving the results of the Kruskal-Wallis test
    kruskal_wallis_results = {}
    kruskal_wallis_results["Feature"] = statistical_test_columns
    kruskal_wallis_results["Statistic"] = kruskal_statistic_list
    kruskal_wallis_results["P-value"] = kruskal_pvalue_list
    kruskal_wallis_results["Interpretation"] = kruskal_pass_fail_list

    kruskal_wallis_results_df = pd.DataFrame.from_dict(kruskal_wallis_results)
    kruskal_wallis_results_df.to_csv(kruskal_path_root+"kruskal_wallis_results_"+kruskal_path_addition+".csv", index=False)

def iqr_outlier_removal(keys, dictionary):
    # Purpose of function: 
    # Performing IQR outlier removal on all columns of a given dictionary (for each key)
    #
    # Inputs:
    # keys = Dictionary keys
    # dictionary = A dictionary of data with keys in the format dis_+number or prox_+number.
    #
    # Outputs:
    # dictionary = A dictionary once IQR outlier removal has been performed.

    # Iterating over each key
    for key in keys:
        # Isolating the key data
        key_data_df = dictionary[key]
        
        columns = list(key_data_df.columns)
        columns = [column for column in columns if column not in ['IICP', 'Patient']]
        
        # Iterating over each column
        for column in columns:
            # Isolating the data of a specific column
            column_data = key_data_df[column]
            
            # Calculating the 25th percentile, 75th percentile and the interquartile range
            percentile_25 = np.percentile(column_data.dropna(),25)
            percentile_75 = np.percentile(column_data.dropna(),75)
            iqr = (percentile_75 - percentile_25)

            # Defining a upper and lowe bound for outlier detection
            lower_bound = (percentile_25 - (1.5 * iqr))
            upper_bound = (percentile_75 + (1.5 * iqr))

            # Removing all data in the current column who's value is < lower bound and > upper bound
            key_data_df.drop(key_data_df[key_data_df[column] < lower_bound].index, inplace=True)
            key_data_df.drop(key_data_df[key_data_df[column] > upper_bound].index, inplace=True)
        
        # Saving data once IQR outlier removal has been performed
        dictionary[key] = key_data_df
    
        return dictionary

def run_wavelength_analysis(dis_prox = False, prox_only = False, dis_only = False, **kwargs):
    keys_to_use = filter_keys(kwargs, dis_prox = dis_prox, prox_only = prox_only, dis_only = dis_only)
    
    # using now() to get current time
    current_time = datetime.datetime.now()
    current_date_time = str(current_time.day) + "_" + str(current_time.month) + "_" + str(current_time.year) + "@" + str(current_time.hour)

    path_root = "Wavelength analysis " + current_date_time +"/"

    kruskal_path_addition = " "
    if dis_prox == True:
        kruskal_path_addition = "distal_proximal"
    elif prox_only == True:
        kruskal_path_addition = "proximal"
    elif dis_only == True:
        kruskal_path_addition = "distal"

    iqr_outlier_removal(keys_to_use, kwargs)
    normality_tests(keys_to_use, path_root, kwargs)
    kruskal_wallis(kruskal_path_addition, keys_to_use, path_root, kwargs)


