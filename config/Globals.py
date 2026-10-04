'''
Created on Mar 8, 2022

@author: Hazegoblin
'''
import os

class Config(object):
    VALUE_LIMIT = 10000000
    ITERATION_LIMIT = 10
    main_dir = os.path.split(os.path.abspath(__file__))[0]
    GRID_WIDTH = 100
    GRID_HEIGHT = 100
    INITIAL_POPULATION = 90
    WINDOW_WIDTH = 1200
    WINDOW_HEIGHT = 820
    X = WINDOW_WIDTH
    Y = WINDOW_HEIGHT
    