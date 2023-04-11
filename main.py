import pandas as pd
from feature_extraction_code.sqi_extraction import get_data_sqis

# Use this file to call upon other files and functions for execution
test_data = pd.read_csv("test_data.csv")
get_data_sqis(test_data, 100, 6000, "test", visualise=0)





