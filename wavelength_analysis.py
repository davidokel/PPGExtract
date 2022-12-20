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
import pandas as pd
from sklearn import preprocessing
import seaborn as sns

distal_features = load_csv("Features_All_Wav/770/distal_features.csv")
proximal_features = load_csv("Features_All_Wav/770/proximal_features.csv")

run_boxplots = True
run_mann = True
run_kruskal = True
run_norm_test = True

def normality_tests(data):
    print("Test for normality")

def mann_whitney(data):
    print("Test for mann whitney")

def kruskal_wallis(data):
    print("Test for Kruskal Wallis")


