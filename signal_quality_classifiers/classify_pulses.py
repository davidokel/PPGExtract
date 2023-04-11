from signal_quality_classifiers.dictionary_formatter import dict_to_df
import pickle
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def get_pulse_predictions(dict, model_path, debug = False):
   # Convert dict to df
   df = dict_to_df(dict)
   # Remove the class column from the dataframe
   df = df.drop(['Peak', 'Pre_peak', 'Post_peak', 'Relative_peak', 'raw_pulse_data', 'norm_pulse_data', 'class'], axis=1)
   
   scaler = StandardScaler()
   df = scaler.fit_transform(df)
   
   classifier_pkl = open(model_path, 'rb')
   classifier = pickle.load(classifier_pkl)

   # Get the prediction for every row in the dataframe
   predictions = classifier.predict(df)

   # If the prediction is 0 then the pulse is "good" and if it is 1 then the pulse is "poor"
   # Given the dictionary and the predictions, change the value "class" to either "good" or "poor" for each key in the dictionary
   
   # Iterating over the dictionary
   for i, key in enumerate(dict.keys()):
      if predictions[i] == 0:
         dict[key]['class'] = 'good'
      else:
         dict[key]['class'] = 'poor'

   if debug == True:
      # Looping over the dictionary and plotting the 'norm_pulse_data' for each pulse along with the prediction
      for key in dict.keys():
         plt.plot(dict[key]['norm_pulse_data'])
         plt.title(dict[key]['class'])
         plt.show()

   return dict
