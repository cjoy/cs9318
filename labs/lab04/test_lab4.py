from submission import multinomial_nb
import pandas as pd
import numpy as np

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

# The basic test provided
def test_1():
    '''
    >>> training_data = test_1()
    >>> sms = 'I am not spam'
    >>> mnb = multinomial_nb(training_data, tokenize(sms))
    >>> np.isclose(mnb, 0.2342767295597484)
    True
    '''
    raw_data = pd.read_csv('./asset/data.txt', sep='\t')
    training_data = []
    for index in range(len(raw_data)):
        training_data.append((get_freq_of_tokens(raw_data.iloc[index].text), raw_data.iloc[index].category))
    return training_data

# Larger test case
def test_2():
    '''
    >>> training_data = test_2()
    >>> sms = 'Free sexy singles, text this number'
    >>> mnb = multinomial_nb(training_data, tokenize(sms))
    >>> np.isclose(mnb, 29.758455873583703)
    True

    >>> sms = 'Normal sms, not spam'
    >>> mnb = multinomial_nb(training_data, tokenize(sms))
    >>> np.isclose(mnb, 0.06500061054165997)
    True

    >>> sms = 'URGENT! text me answer to Q3 free'
    >>> mnb = multinomial_nb(training_data, tokenize(sms))
    >>> np.isclose(mnb, 24.48157639967678)
    True

    >>> sms = 'hi man, hows it hangin'
    >>> mnb = multinomial_nb(training_data, tokenize(sms))
    >>> np.isclose(mnb, 0.00040916253796595637)
    True
    '''
    raw_data = pd.read_csv('./asset/spam.csv', sep=',', encoding='latin-1')
    training_data = []
    for index in range(len(raw_data)):
        training_data.append((get_freq_of_tokens(raw_data.iloc[index].v2), raw_data.iloc[index].v1))
    return training_data

if __name__ == '__main__':
    import doctest
    doctest.testmod()
