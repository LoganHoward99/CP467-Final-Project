import cv2
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
import os

# -----------------------
# Paramters
# -----------------------
NUM_SCENES = 5
NUM_OBJECTS = 15

# FLANN
FLANN_INDEX_KDTREE = 1
FLANN_INDEX_PARAMS = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
FLANN_SEARCH_PARAMS = dict(checks=50)

# Ratio Test
MATCH_TH = 0.75


# Object Dictionary - Maps object image to object name
objectDict = {
    'O1': 'Dawn Spry',
    'O2': 'Water Bottle',
    'O3': 'Spray Bottle',
    'O4': 'Candle',
    'O5': 'Book',
    'O6': 'Pliers',
    'O7': 'Paint Set',
    'O8': 'Strainer',
    'O9': 'Cup',
    'O10': 'Pot',
    'O11': 'Decoration',
    'O12': 'Lego Flower',
    'O13': 'Bowl',
    'O14': 'Axe',
    'O15': 'Ornament',
}

# Scene Dictionary - Maps scene to objects in scene
sceneDict = {
    'S1': ['O1', 'O2', 'O3', 'O4', 'O5', 'O6', 'O7', 'O8', 'O9', 'O10', 'O11', 'O12', 'O13', 'O14', 'O15'],
    'S2': ['O1', 'O2', 'O4', 'O5', 'O6', 'O7', 'O8', 'O9', 'O11', 'O12', 'O13', 'O14', 'O15'],
    'S3': ['O1', 'O2', 'O3', 'O5', 'O6', 'O7', 'O8', 'O9', 'O10', 'O11', 'O12', 'O13', 'O14'],
    'S4': ['O1', 'O6', 'O7', 'O8', 'O9', 'O10', 'O11', 'O12', 'O13', 'O14'],
    'S5': ['O1', 'O6', 'O7', 'O8', 'O9', 'O10', 'O11', 'O12', 'O14'],
    # 'S6': ['O1', 'O6', 'O7', 'O8', 'O9', 'O11', 'O12', 'O13', 'O14'],
    # 'S7': ['O2', 'O5', 'O7', 'O8', 'O11', 'O14', 'O15'],
    # 'S8': ['O2', 'O4', 'O5', 'O7', 'O8', 'O14', 'O15'],
    # 'S9': ['O1', 'O6', 'O7', 'O8', 'O9', 'O10', 'O11', 'O12', 'O13', 'O14'],
    # 'S10': ['O2', 'O3', 'O4', 'O5', 'O7', 'O8', 'O15'],
    # 'S11': ['O2', 'O3', 'O4', 'O5', 'O15'],
    # 'S12': ['O2', 'O3', 'O4', 'O5', 'O7', 'O8', 'O13', 'O14', 'O15'],
    # 'S13': ['O1', 'O2', 'O3', 'O4', 'O5', 'O7', 'O8', 'O11', 'O13', 'O14', 'O15'],
    # 'S14': ['O1', 'O2', 'O4', 'O5', 'O7', 'O8', 'O9', 'O11', 'O12', 'O13', 'O14', 'O15'],
    # 'S15': ['O1', 'O2', 'O5', 'O6', 'O7', 'O8', 'O9', 'O11', 'O12', 'O13', 'O14', 'O15'],
    # 'S16': ['O1', 'O2', 'O6', 'O7', 'O8', 'O9', 'O10', 'O11', 'O12', 'O13', 'O14'],
    # 'S17': ['O1', 'O6', 'O7', 'O8', 'O9', 'O10', 'O11', 'O12', 'O14'],
    # 'S18': ['O1', 'O6', 'O7', 'O8', 'O9', 'O10', 'O11'],
    # 'S19': ['O1', 'O6', 'O7', 'O8', 'O9', 'O10'],
    # 'S20': ['O1', 'O7', 'O8', 'O9'],
    # 'S21': ['O1', 'O7', 'O8'],
    # 'S22': ['O1', 'O2', 'O7', 'O8', 'O11', 'O14', 'O15'],
    # 'S23': ['O2', 'O4', 'O7', 'O8', 'O15'],
    # 'S24': ['O2', 'O3', 'O4', 'O5', 'O7', 'O8', 'O15'],
    # 'S25': ['O2', 'O3', 'O4', 'O5', 'O15'],
    # 'S26': ['O3', 'O4'],
    # 'S27': ['O2', 'O3', 'O4', 'O5', 'O7', 'O8', 'O13', 'O14', 'O15'],
    # 'S28': ['O1', 'O2', 'O4', 'O5', 'O6', 'O7', 'O8', 'O9', 'O11', 'O12', 'O13', 'O14', 'O15'],
    # 'S29': ['O1', 'O5', 'O6', 'O7', 'O8', 'O9', 'O10', 'O11', 'O12', 'O13', 'O14'],
    # 'S30': ['O1', 'O2', 'O3', 'O4', 'O5', 'O6', 'O7', 'O8', 'O9', 'O10', 'O11', 'O12', 'O13', 'O14', 'O15'],
}

