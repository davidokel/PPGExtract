import pandas as pd
from protocols import extraction_protocol

# Use this file to call upon other files and functions for execution
test_dataset = pd.read_csv("data/test_data.csv")
test_data = test_dataset["Data"]
test_data = test_data * -1
extraction_protocol(test_dataset, test_data, 100, 6000, "test", visualise=1, debug=0)

