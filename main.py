import pandas as pd
from feature_extraction_code.protocols import extraction_protocol

# Use this file to call upon other files and functions for execution
test_data = pd.read_csv("Data/test_data.csv")
extraction_protocol(test_data, 100, 6000, "test", visualise=0, debug=1)





