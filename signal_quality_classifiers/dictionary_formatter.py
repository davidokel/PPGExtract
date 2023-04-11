import pandas as pd

def dict_to_df(dictionary):
    keys = list(dictionary.keys())
    keys.reverse()
    instance_index = next((key for key in keys if dictionary[key]["class"] == "poor"), keys[-1]) + 1

    print(instance_index)

    # Create an empty DataFrame
    df = pd.DataFrame()

    # Loop through the keys of the dictionary and add each value as a column in the DataFrame
    for key in dictionary.keys():
        if key < instance_index:
            row_data = dictionary[key]
            row_dict = {}
            for k in row_data.keys():
                if k == 'features':
                    row_dict.update(row_data[k])
                else:
                    row_dict[k] = row_data[k]
            df = df.append(row_dict, ignore_index=True)
        else:
            break

    # Print the resulting DataFrame
    return df
