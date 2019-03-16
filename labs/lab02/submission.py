## import modules here 
import pandas as pd
import numpy as np



################# Question 1 #################

def v_opt_dp(x, num_bins):# do not change the heading of the function
  bins = pd.cut(np.array(x), bins=num_bins)
  return bins, []


################# Delete me #################
x = [3, 1, 18, 11, 13, 17]
num_bins = 4
bins, matrix = v_opt_dp(x, num_bins)
print("Bins = {}".format(bins))
print("Matrix =")
for row in matrix:
  print(row)




# Output:
#
# Bins = [[3, 1], [18], [11, 13], [17]]
# Matrix =
# [-1, -1, -1, 18.666666666666664, 8.0, 0]
# [-1, -1, 18.666666666666664, 2.0, 0, -1]
# [-1, 18.666666666666664, 2.0, 0, -1, -1]
# [4.0, 2.0, 0, -1, -1, -1]
