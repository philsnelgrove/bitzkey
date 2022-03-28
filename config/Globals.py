'''
Created on Mar 8, 2022

@author: Hazegoblin
'''
import os

class Config(object):
    VALUE_LIMIT = 10000000
    ITERATION_LIMIT = 10
    main_dir = os.path.split(os.path.abspath(__file__))[0]
    X = 1024
    Y = 768
    