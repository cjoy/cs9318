## import modules here 
import pandas as pd
from numpy import product

def tokenize(sms):
  return sms.split(' ')

def get_freq_of_tokens(sms):
  tokens = {}
  for token in tokenize(sms):
    if token not in tokens:
      tokens[token] = 1
    else:
      tokens[token] += 1
  return tokens

################# Question 1 #################
def multinomial_nb(training_data, sms):# do not change the heading of the function
  # Meta for later calculations
  vocabulary = set([k for t in training_data for k in t[0]])

  # Prior probability given class: P(C) = N_C / N
  probability_class = lambda klass : sum(1 for t in training_data if t[1] == klass) / len(training_data)

  # Conditional probability ~ word given class: P(w|C) = Count(w, c) + 1 / Count(c) + |V|
  count_word_class = lambda word, klass : sum(t[0][word] for t in training_data if t[1] == klass and word in t[0])
  count_class = lambda klass : sum(sum(t[0].values()) for t in training_data if t[1] == klass)
  probability_word_class = lambda word, klass : (count_word_class(word, klass) + 1) / (count_class(klass) + len(vocabulary))

  # Probability of class given sms: probability_class * (probability_word_class for each word)
  probability_class_sms = lambda klass : probability_class(klass) * product([probability_word_class(word, klass) for word in sms])
  
  # Ratio of probability of sms is spam to probability of sms is ham
  return probability_class_sms('spam') / probability_class_sms('ham')

if __name__ == '__main__':
  raw_data = pd.read_csv('./asset/data.txt', sep='\t')
  training_data = []
  for index in range(len(raw_data)):
    training_data.append((get_freq_of_tokens(raw_data.iloc[index].text), raw_data.iloc[index].category))

  sms = 'I am not spam'
  print(multinomial_nb(training_data, tokenize(sms)))