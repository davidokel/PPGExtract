import pandas as pd
from protocols import extraction_protocol_v2
import matplotlib.pyplot as plt
import pickle as pkl
from support_code.pulse_detection import get_pulses

# Column names to keep 
#columns_to_keep = ["1","10","14","15","16","17","18","19","20","22","23","25","26","27","28","29","36","38","4","5","6","7","8"]

#nicp_dataset = pd.read_pickle("D:\\Denoising\\NICP\\nICP_denoised_joint.pkl")
#icp_dataset = pd.read_pickle("D:\\Denoising\\ICP\\ICP_denoised_joint.pkl")

"""# Get index of column with name "22"
index = nicp_dataset.columns.get_loc("27")

# Get data from column 22 onwards and get the top 12000 rows
nicp_dataset = nicp_dataset.iloc[:12000,index:]
icp_dataset = icp_dataset.iloc[:12000,index:]"""

#extraction_protocol_v2(nicp_dataset, icp_dataset, 100, 6000, visualise=False, debug=False, derivative=[1,2])

# Load test_10_06_2023.csv
test_dataset = pd.read_csv("test_10_06_2023.csv")
data = test_dataset["Data"]

# Iterate over the data in windows of 6000 samples
for i in range(0, len(data), 6000):
    # Get the data in the window
    window = data[i:i+6000]
    # Get the pulses in the window
    pulses, peaks, troughs = get_pulses(list(window), fs=100, visualise=True, debug=True, z_score_threshold=3, z_score_detection=True)


