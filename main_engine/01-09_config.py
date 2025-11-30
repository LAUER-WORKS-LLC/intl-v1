"""
Configuration file for Feature Predictor Powerhouse Model Pipeline
"""

import os
from datetime import datetime

# ============================================================================
# BASE PATHS
# ============================================================================
# Base directory (will be set on instance)
BASE_DIR = os.path.join(os.path.expanduser('~'), '09_train_features')

# Data directories (created automatically)
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
NORMALIZED_DATA_DIR = os.path.join(DATA_DIR, 'normalized')
SMOOTHED_DATA_DIR = os.path.join(DATA_DIR, 'smoothed')
KEY_POINTS_DIR = os.path.join(DATA_DIR, 'key_points')
CHUNKS_DIR = os.path.join(DATA_DIR, 'chunks')
SPLINES_DIR = os.path.join(DATA_DIR, 'splines')
FEATURES_DIR = os.path.join(DATA_DIR, 'features')
TRAINING_DATA_DIR = os.path.join(DATA_DIR, 'training')
NORMALIZATION_FACTORS_DIR = os.path.join(DATA_DIR, 'normalization_factors')
MODELS_DIR = os.path.join(DATA_DIR, 'models')
LOGS_DIR = os.path.join(DATA_DIR, 'logs')

# Source data
ALL_STOCKS_JSON = os.path.join(BASE_DIR, 'data', 'all_stocks_polygon_working_20251016_105954.json')

# ============================================================================
# DATA FETCHING
# ============================================================================
START_DATE = '2000-01-01'
END_DATE = '2025-01-01'

# ============================================================================
# NORMALIZATION
# ============================================================================
# Normalize prices by dividing by first price in series
NORMALIZE_PRICES = True
NORMALIZE_VOLUME = True  # Use log transform for volume
VOLUME_LOG_BASE = 10  # Base for log transform

# ============================================================================
# LOWESS SMOOTHING
# ============================================================================
LOWESS_FRAC = 0.25
LOWESS_IT = 3  # Number of iterations
LOWESS_DELTA = 0.0  # Distance for robust estimation

# Validation thresholds
LOWESS_MAX_DEVIATION_STD = 3.0  # Max deviation from raw data (in std devs)
LOWESS_MIN_DATA_POINTS = 50  # Minimum data points required

# ============================================================================
# KEY POINT DETECTION
# ============================================================================
# Minimum distance between key points (in trading days)
MIN_KEY_POINT_DISTANCE = 5

# Key point detection parameters
INFLECTION_CURVATURE_THRESHOLD = 0.001  # Minimum curvature change for inflection
PEAK_PROMINENCE = 0.01  # Minimum prominence for peaks/valleys

# ============================================================================
# CHUNK VALIDATION
# ============================================================================
MIN_CHUNK_DURATION = 3  # Minimum days in a chunk
MAX_CHUNK_DURATION = 200  # Maximum days in a chunk
MAX_CHUNK_RETURN = 0.5  # Maximum return percentage (50%)

# Only create chunks between actual key points (no first_day/last_day)
ONLY_KEY_POINT_CHUNKS = True

# ============================================================================
# SPLINE INTERPOLATION
# ============================================================================
SPLINE_SMOOTHING_FACTOR = None  # None = automatic, or set specific value
SPLINE_DEGREE = 3  # Cubic splines

# Validation thresholds
SPLINE_MAX_OSCILLATION = 0.1  # Max oscillation relative to data range
SPLINE_MIN_POINTS = 5  # Minimum points for spline fitting

# ============================================================================
# FEATURE EXTRACTION
# ============================================================================
# Feature validation thresholds
FEATURE_MAX_VALUE = 1000  # Maximum reasonable feature value
FEATURE_MIN_VALUE = -1000  # Minimum reasonable feature value

# ============================================================================
# TRAINING DATASET
# ============================================================================
# Number of previous time periods to use as input (FAQ: 3 time periods → 1)
NUM_PREVIOUS_CHUNKS = 3  # t-3, t-2, t-1 → t (3 previous chunks)
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Time-series aware splitting (respects chronological order - CRITICAL for time-series)
TIME_SERIES_SPLIT = True

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================
# Note: Actual dimensions are read from metadata.json at runtime
# These are defaults/documentation values
# Actual feature count: 66 (geometric_shape_, derivative_, pattern_, transition_)
# Input: 3 previous chunks * 66 features = 198 features
INPUT_DIM = NUM_PREVIOUS_CHUNKS * 66  # 198 (actual from metadata.json)
# Output: 66 features for next chunk
OUTPUT_DIM = 66  # 66 features to predict (actual from metadata.json)

# Network architecture
HIDDEN_LAYERS = [2048, 1024, 512, 256, 128, 64]
DROPOUT_RATES = [0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
USE_BATCH_NORM = True
USE_RESIDUAL_CONNECTIONS = True

# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================
BATCH_SIZE = 512  # FAQ: mini-batch training (32-1024 depending on hardware)
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 500
EARLY_STOPPING_PATIENCE = 30
EARLY_STOPPING_MIN_DELTA = 1e-6

# Shuffle training data (FAQ: mini-batch training, but for time-series, shuffle=False may be better)
# Since we already split chronologically, shuffling within training set is acceptable
SHUFFLE_TRAINING_DATA = True

# Learning rate scheduling (FAQ: reduce LR on plateau)
LR_SCHEDULER = 'plateau'  # 'cosine', 'step', 'plateau', or None
LR_MIN = 1e-6
LR_T_MAX = NUM_EPOCHS  # For cosine annealing
LR_PLATEAU_PATIENCE = 10  # For plateau scheduler
LR_PLATEAU_FACTOR = 0.5  # For plateau scheduler

# Gradient clipping
GRADIENT_CLIP_VALUE = 1.0

# ============================================================================
# LOGGING
# ============================================================================
LOG_LEVEL = 'INFO'
LOG_TO_FILE = True
LOG_FILE = os.path.join(LOGS_DIR, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

# ============================================================================
# VALIDATION
# ============================================================================
# Enable validation at each step
VALIDATE_RAW_DATA = True
VALIDATE_NORMALIZED_DATA = True
VALIDATE_SMOOTHED_DATA = True
VALIDATE_KEY_POINTS = True
VALIDATE_CHUNKS = True
VALIDATE_SPLINES = True
VALIDATE_FEATURES = True
VALIDATE_TRAINING_DATA = True

# ============================================================================
# PROCESSING OPTIONS
# ============================================================================
# Process all stocks or subset
PROCESS_ALL_STOCKS = True
MAX_STOCKS_TO_PROCESS = None  # None = all, or set number

# Parallel processing
USE_MULTIPROCESSING = False  # Set to True for parallel stock processing
NUM_WORKERS = 4  # Number of parallel workers

# Progress reporting
SHOW_PROGRESS = True
SAVE_INTERMEDIATE_RESULTS = True  # Save results after each step

