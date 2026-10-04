'''
Created on Mar 8, 2022

@author: Hazegoblin
'''
import os
import sys
from pathlib import Path

class Config(object):
    VALUE_LIMIT = 10000000
    ITERATION_LIMIT = 10
    main_dir = os.path.split(os.path.abspath(__file__))[0]
    # Where the app writes reports and exports: beside Bitzkey.exe when packaged (a one-file
    # executable runs from a temporary folder), otherwise the project folder.
    APP_DIR = (Path(sys.executable).resolve().parent if getattr(sys, "frozen", False)
               else Path(__file__).resolve().parent.parent)
    GRID_WIDTH = 100
    GRID_HEIGHT = 100
    INITIAL_POPULATION = 90
    GRASS_DENSITY = 5
    FULLSCREEN = True  # start fullscreen; F11 or --windowed for a window of the size below
    WINDOW_WIDTH = 1200
    WINDOW_HEIGHT = 820
    X = WINDOW_WIDTH
    Y = WINDOW_HEIGHT
    