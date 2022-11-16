import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_methods import load_csv
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np

# PURPOSE OF SCRIPT:
# The univariate and multivariate data exploration/analysis of the features extracted from the data.
# This analysis will serve both as a representation of the data for further understanding of each feature and the possible relationships between features.
# The outputs of this analysis will also serve as a basis to compare the datasets before and after selection based on SQIs.

def univariate_analysis(dataframe,folder_save_path):
    # PURPOSE OF FUNCTION:
    # Plot a histogram and density plot for each feature and save them into a folder.
    # Calculate the count, mean, median, mode, std, median absolute deviation, min, max, 25%, 50% and 75% for each feature and for each feature and save them into a csv file.

    # INPUTS:
    # dataframe: The dataframe containing the features.
    # folder_save_path: The path to the folder where the plots will be saved.
    # include_all_columns: If True, all columns will be included in the analysis. If False, only the columns containing the features will be included.

    # OUTPUTS:
    # None

    # CREATING FOLDER TO SAVE PLOTS
    if not os.path.exists(folder_save_path):
        os.makedirs(folder_save_path)
    
    # CREATING DATAFRAME TO SAVE STATISTICS
    statistics = pd.DataFrame(columns = ["Count", "Mean", "Median", "Mode", "Std", "Median Absolute Deviation", "Min", "Max", "25%", "50%", "75%"])

    # CREATING PLOTS AND STATISTICS
    for column in dataframe.columns:
        # CREATING PLOTS
        fig, ax = plt.subplots(1,2, figsize = (20,10))
        sns.distplot(dataframe[column], ax = ax[0])
        sns.histplot(dataframe[column], ax = ax[1])
        fig.suptitle(column, fontsize = 20)
        fig.savefig(folder_save_path + "/" + column + ".png")
        plt.close(fig)

        # CREATING STATISTICS
        statistics.loc[column] = [dataframe[column].count(), dataframe[column].mean(), dataframe[column].median(), dataframe[column].mode()[0], dataframe[column].std(), dataframe[column].mad(), dataframe[column].min(), dataframe[column].max(), dataframe[column].quantile(0.25), dataframe[column].quantile(0.5), dataframe[column].quantile(0.75)]
    
    # SAVING STATISTICS
    statistics.to_csv(folder_save_path + "/Statistics.csv")

def multivariate_analysis(dataframe, folder_save_path):
    # PURPOSE OF FUNCTION:
    # Plot a scatter plot for each pair of features and save them into a folder.
    # Calculate the correlation coefficient and plot a heatmap for all features and save them into a csv file.
    # Calculate the covariance matrix and plot a heatmap for all features and save them into a csv file.
    
    # INPUTS:
    # dataframe: The dataframe containing the features.
    # folder_save_path: The path to the folder where the plots will be saved.

    # OUTPUTS:
    # None

    # CREATING FOLDER TO SAVE PLOTS
    if not os.path.exists(folder_save_path):
        os.makedirs(folder_save_path)

    """# CREATING SNS PAIRPLOTS AND STATISTICS
    sns.pairplot(dataframe)
    plt.savefig(folder_save_path + "/Pairplot.png")
    plt.close()"""

    # CREATING SCATTER PLOTS
    # Creating a folder for all scatter plots using mkdir
    if not os.path.exists(folder_save_path + "/Scatter Plots"):
        os.makedirs(folder_save_path + "/Scatter Plots")
    # Creating a scatter plot for each pair of features
    for column1 in dataframe.columns:
        for column2 in dataframe.columns:
            if column1 != column2:
                fig, ax = plt.subplots(figsize = (20,10))
                sns.scatterplot(data = dataframe, x = column1, y = column2, ax = ax)
                fig.suptitle(column1 + " vs " + column2, fontsize = 20)
                fig.savefig(folder_save_path + "/Scatter Plots/" + column1 + "_vs_" + column2 + ".png")
                plt.close(fig)

    # CREATING HEATMAPS, Pearsons, Kendall and Spearman
    correlation_coefficients = dataframe.corr(method = "pearson")
    sns.heatmap(correlation_coefficients, annot = True)
    # Make fig large
    fig = plt.gcf()
    fig.set_size_inches(20, 10)
    plt.savefig(folder_save_path + "/Pearsons.png")
    plt.close()

    correlation_coefficients = dataframe.corr(method = "kendall")
    sns.heatmap(correlation_coefficients, annot = True)
    fig = plt.gcf()
    fig.set_size_inches(20, 10)
    plt.savefig(folder_save_path + "/Kendall.png")
    plt.close()

    correlation_coefficients = dataframe.corr(method = "spearman")
    sns.heatmap(correlation_coefficients, annot = True)
    fig = plt.gcf()
    fig.set_size_inches(20, 10)
    plt.savefig(folder_save_path + "/Spearman.png")
    plt.close()

    # CREATING COVARIANCE MATRIX
    covariance_matrix = dataframe.cov()
    sns.heatmap(covariance_matrix, annot = True)
    fig = plt.gcf()
    fig.set_size_inches(20, 10)
    plt.savefig(folder_save_path + "/Covariance.png")
    plt.close()

# IMPORTING FEATURE DATA
distal_features = load_csv("Features/Joint_Features/Extraction_WITH_SQIS_DISTAL_NORM.csv")
proximal_features = load_csv("Features/Joint_Features/Extraction_WITH_SQIS_PROXIMAL_NORM.csv")
subtracted_features = load_csv("Features/Joint_Features/Extraction_WITH_SQIS_SUBTRACTED_NORM.csv")

univariate_analysis(distal_features, "Features/Joint_Features/Univariate_Analysis/Distal")
univariate_analysis(proximal_features, "Features/Joint_Features/Univariate_Analysis/Proximal")
univariate_analysis(subtracted_features, "Features/Joint_Features/Univariate_Analysis/Subtracted")

multivariate_analysis(distal_features, "Features/Joint_Features/Multivariate_Analysis/Distal")
multivariate_analysis(proximal_features, "Features/Joint_Features/Multivariate_Analysis/Proximal")
multivariate_analysis(subtracted_features, "Features/Joint_Features/Multivariate_Analysis/Subtracted")





