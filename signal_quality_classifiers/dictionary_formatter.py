import pandas as pd

def dict_to_df(dictionary):
    keys = list(dictionary.keys())
    keys.reverse()

    # Create an empty DataFrame
    df = pd.DataFrame()

    # Loop through the keys of the dictionary and add each value as a column in the DataFrame
    rows = []
    for key in dictionary.keys():
        row_data = dictionary[key]
        row_dict = {}
        for k in row_data.keys():
            if k == 'features':
                row_dict.update(row_data[k])
            else:
                row_dict[k] = row_data[k]
        rows.append(row_dict)

    df = pd.concat([df, pd.DataFrame(rows)])

    # Print the resulting DataFrame
    return df.reset_index(drop=True)
