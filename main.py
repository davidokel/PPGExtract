import pandas as pd
from protocols import extraction_protocol_v2
import matplotlib.pyplot as plt

# Use this file to call upon other files and functions for execution
test_dataset = pd.read_csv("data/test_data.csv")
test_data = test_dataset["Data"]
test_data = test_data * -1

extraction_protocol_v2(test_dataset, test_data, 100, 6000, "test", visualise=True, debug=True, derivative=0)


