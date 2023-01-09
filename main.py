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
from protocol_updated import *
from wavelength_analysis import run_wavelength_analysis

dis_770 = pd.read_csv("Features_All_Wav/770/distal_features.csv")
dis_810 = pd.read_csv("Features_All_Wav/810/distal_features.csv")
dis_850 = pd.read_csv("Features_All_Wav/850/distal_features.csv")
dis_880 = pd.read_csv("Features_All_Wav/880/distal_features.csv")

prox_770 = pd.read_csv("Features_All_Wav/770/proximal_features.csv")
prox_810 = pd.read_csv("Features_All_Wav/810/proximal_features.csv")
prox_850 = pd.read_csv("Features_All_Wav/850/proximal_features.csv")
prox_880 = pd.read_csv("Features_All_Wav/880/proximal_features.csv")

run_wavelength_analysis(dis_only = True, dis_770=dis_770, dis_810=dis_810, dis_850=dis_850, dis_880=dis_880)
run_wavelength_analysis(prox_only = True, prox_770=prox_770, prox_810=prox_810, prox_850=prox_850, prox_880=prox_880)


