import pandas as pd
from protocols import extraction_protocol_v2
from support_code.pulse_detection import get_pulses

# Load test_10_06_2023.csv
test_dataset = pd.read_csv("test_10_06_2023.csv")
data = test_dataset["Data"]

# Iterate over the data in windows of 6000 samples
for i in range(0, len(data), 6000):
    # Get the data in the window
    window = data[i:i+6000]
    # Get the pulses in the window
    pulses, peaks, troughs = get_pulses(list(window), fs=100, visualise=True, debug=True, z_score_threshold=3, z_score_detection=True)


