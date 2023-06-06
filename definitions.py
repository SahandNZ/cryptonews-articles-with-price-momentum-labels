import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
CLEAN_DATA_DIR = os.path.join(DATA_DIR, 'clean')
TEXTUAL_RAW_DATA_DIR = os.path.join(RAW_DATA_DIR, 'textual')
NUMERICAL_RAW_DATA_DIR = os.path.join(RAW_DATA_DIR, 'numerical')
TEXTUAL_CLEAN_DATA_DIR = os.path.join(CLEAN_DATA_DIR, 'textual')
NUMERICAL_CLEAN_DATA_DIR = os.path.join(CLEAN_DATA_DIR, 'numerical')
