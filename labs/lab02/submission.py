# import modules here 
import pandas as pd
import numpy as np


################# Question 1 #################
def create_inital_matrix(x, num_bins):
  return [[-1 for b in range(len(x))] for a in range(num_bins)]

def sse(arr):
  if len(arr) == 0: return 0
  E = np.mean(arr)
  return np.sum([(x-E)**2 for x in arr])



bins = {}

def v_opt_dp_util(x, num_bins, curr_bin, tab, compute):

  remain_bins = num_bins - curr_bin

  scores = {}

  for b in range(1, len(x)):
    head = x[:b]
    tail = x[b:]

    print('\t'*tab, head, curr_bin, ' - ', tail, remain_bins)

    if curr_bin == 1:
      print('\t'*tab,'sse head - ', sse(head))
      compute[curr_bin][b-1] = sse(head)
      # if str(head) not in scores: scores[str(head)] = [sse(head)]
      # else: scores[str(head)].append(sse(head))
      scores[str(head)] = sse(head)

    if remain_bins == 1:
      print('\t'*tab,'sse tail - ', sse(tail))
      compute[curr_bin][b-1] = sse(tail)
      # if str(tail) not in scores: scores[str(tail)] = [sse(tail)]
      # else: scores[str(tail)].append(sse(tail))
      scores[str(tail)] = sse(tail)

 
    if remain_bins > 1:
      v_opt_dp_util(tail, num_bins, curr_bin+1, tab+1, compute)

  if len(scores) > 0:
    min_key = min(scores, key=scores.get)
    min_score = scores[min_key]
    print('\t'*tab,'min_key',min_key, 'min_score', min_score, 'total_cost', 0, '\t', 'scores -', scores)
    print()
    bins[curr_bin] = min_key

  print()

  # v_opt_dp(x[1:], num_bins - 1)

  # print(bins)

def v_opt_dp(x, num_bins):
  compute = create_inital_matrix(x, num_bins)
  v_opt_dp_util(x, num_bins, 1, 0, compute)

  for row in compute:
    print(row)

  print('bins = ', bins)

if __name__ == '__main__':
  x = [3, 1, 18, 11, 13, 17]
  num_bins = 4
  v_opt_dp(x, num_bins)

  # x = [7, 9, 13, 5]
  # num_bins = 3
  # v_opt_dp(x, num_bins)

  # x = [3, 1, 18, 11, 13, 17]
  # num_bins = 4
  # v_opt_dp(x, num_bins)
  # matrix, bins = v_opt_dp(x, num_bins)
  # print('bins = ', bins)
  # print('matrix = ')
  # for row in matrix:
  #   print(row)