'''
Created on Mar 8, 2022

@author: Hazegoblin
'''
import os
import sys
from pathlib import Path

class Config(object):
    VERSION = "1.63.0"  # shown at the top of every screen
    CORES = 1  # the processor cores the simulation thinks on: Python runs it on one
    VALUE_LIMIT = 10000000
    ITERATION_LIMIT = 10
    main_dir = os.path.split(os.path.abspath(__file__))[0]
    # Where the app writes reports and exports: beside Bitzkey.exe when packaged (a one-file
    # executable runs from a temporary folder), otherwise the project folder.
    APP_DIR = (Path(sys.executable).resolve().parent if getattr(sys, "frozen", False)
               else Path(__file__).resolve().parent.parent)
    GRID_WIDTH = 360  # a new world twice as wide as it is tall, as a globe has it
    GRID_HEIGHT = 180
    INITIAL_POPULATION = 300
    GRASS_DENSITY = 50
    OCEAN = 33  # percent of a new world's map under the sea
    MIND_MUTATION = 50  # percent chance a birth mutates the child's mind
    BODY_MUTATION = 50  # and its body
    YEAR_LENGTH = 2880  # turns in a year: 360 days of 8 turns
    FULLSCREEN = True  # start fullscreen; F11 or --windowed for a window of the size below
    WINDOW_WIDTH = 1200
    WINDOW_HEIGHT = 820
    X = WINDOW_WIDTH
    Y = WINDOW_HEIGHT
    
