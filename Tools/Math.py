'''
Created on Mar 8, 2022

@author: Hazegoblin
'''

VALUE_LIMIT = 10000000

def fibonacci(count, old, new, seq):
    newer = old + new
    if newer < VALUE_LIMIT:
        count += 1
        seq.append(newer)
        fibonacci(count, new, newer, seq)
    else:
        seq.append(f"(The next item would be {newer})")
        seq.append(f"{count} steps total.")
    return seq

class Math(object):
    '''
    classdocs
    '''


    def __init__(self, params):
        '''
        Constructor
        '''
        