import numpy as np
import pandas as pd


def logistic_regression(data, labels, weights, num_epochs, learning_rate): # do not change the heading of the function
  pass




if __name__ == '__main__':
  data_file='./asset/a'
  ## Read in the Data...
  raw_data = pd.read_csv(data_file, sep=',')
  labels=raw_data['Label'].values
  data=np.stack((raw_data['Col1'].values,raw_data['Col2'].values), axis=-1)

  np.set_printoptions(threshold=np.nan)
  print(labels)
  # print(data)

  ## Fixed Parameters. Please do not change values of these parameters...
  weights = np.zeros(3) # We compute the weight for the intercept as well...
  num_epochs = 50000
  learning_rate = 50e-5

  coefficients=logistic_regression(data, labels, weights, num_epochs, learning_rate)
  print(coefficients)