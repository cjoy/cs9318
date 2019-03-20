## import modules here 
import pandas as pd
import numpy as np

################# Question 1 #################
def sse(arr):
  E = np.mean(arr)
  return np.sum([(x-E)**2 for x in arr])

def v_opt_dp_util(x, num_bins, matrix, scores, matrix_x, remaining_bins):
  if (num_bins - remaining_bins - matrix_x >= 2) or (len(x) - matrix_x <= remaining_bins):
    return

  v_opt_dp_util(x, num_bins, matrix, scores, matrix_x + 1, remaining_bins)

  if (remaining_bins == 0):
    matrix[remaining_bins][matrix_x] = np.var(x[matrix_x:]) * len(x[matrix_x:])
    return

  v_opt_dp_util(x, num_bins, matrix, scores, matrix_x, remaining_bins - 1)
  matrix_l = [matrix[remaining_bins - 1][matrix_x + 1]]
  matrix_l.extend( [matrix[remaining_bins - 1][i] + (i - matrix_x) * np.var(x[matrix_x:i]) for i in range(matrix_x + 2, len(x))])
  matrix[remaining_bins][matrix_x] = min(matrix_l)
  scores[remaining_bins][matrix_x] = matrix_l.index(min(matrix_l)) + matrix_x + 1


def v_opt_dp(x, num_bins):
  matrix = [[-1 for i in range(len(x))] for j in range(num_bins)]
  scores = [[-1 for i in range(len(x))] for j in range(num_bins)]

  v_opt_dp_util(x, num_bins, matrix, scores, 0, num_bins - 1)

  nxt = scores[-1][0]
  prev = nxt
  bins = [x[:nxt]]

  for i in range(len(scores) - 2, 0, -1):
    nxt = scores[i][nxt]
    bins.append(x[prev:nxt])
    prev = nxt
  bins.append(x[prev:])

  return matrix, bins

################# Stuff for testing #################
# x = [3, 1, 18, 11, 13, 17]
# num_bins = 4
# matrix, bins = v_opt_dp(x, num_bins)
# print('bins = ', bins)
# print('matrix = ')
# for row in matrix:
#   print(row)

# x = [7, 9, 13, 5]
# num_bins = 4
# v_opt_dp(x, num_bins)



# The input contains data (in a list) and the number of bins (an integer).
# You are required to output the binning result and the matrix computed by the algorithm.
#
# The matrix entries record optimal binning cost for a suffix of the input array using a certain number of bins. You should assign -1 to all the invalid solutions.
#
# Output:
#
# Bins = [[3, 1], [18], [11, 13], [17]]
# Matrix =
# [-1, -1, -1, 18.666666666666664, 8.0, 0]
# [-1, -1, 18.666666666666664, 2.0, 0, -1]
# [-1, 18.666666666666664, 2.0, 0, -1, -1]
# [4.0, 2.0, 0, -1, -1, -1]
#
