## import modules here 
import pandas as pd
from numpy import product

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
  probability_class_sms = lambda klass : probability_class(klass) * product([probability_word_class(word, klass) for word in sms if word in vocabulary])
  # Ratio of probability of sms is spam to probability of sms is ham
  return probability_class_sms('spam') / probability_class_sms('ham')
