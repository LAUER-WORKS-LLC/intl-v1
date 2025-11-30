"""
Configuration file for Second-Stage Models (Canonical Path + OHLCV Residual)
"""

import os

# ============================================================================
# BASE PATHS
# ============================================================================
# Reference to first-stage pipeline
FIRST_STAGE_DIR = os.path.join(os.path.expanduser('~'), '09_train_features')
FIRST_STAGE_DATA_DIR = os.path.join(FIRST_STAGE_DIR, 'data')
FIRST_STAGE_TRAINING_DIR = os.path.join(FIRST_STAGE_DATA_DIR, 'training')
FIRST_STAGE_MODELS_DIR = os.path.join(FIRST_STAGE_DATA_DIR, 'models')

# Second-stage base directory
BASE_DIR = os.path.join(os.path.expanduser('~'), 'train_second_models')
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Step-specific directories
STEP_10_DIR = os.path.join(BASE_DIR, 'step_10')
STEP_11_DIR = os.path.join(BASE_DIR, 'step_11')
STEP_12_DIR = os.path.join(BASE_DIR, 'step_12')
STEP_13_DIR = os.path.join(BASE_DIR, 'step_13')

# Canonical path dataset
CANONICAL_DATA_DIR = os.path.join(DATA_DIR, 'canonical')
CANONICAL_TRAINING_DIR = os.path.join(CANONICAL_DATA_DIR, 'training')

# OHLCV residual dataset
RESIDUAL_DATA_DIR = os.path.join(DATA_DIR, 'residual')
RESIDUAL_TRAINING_DIR = os.path.join(RESIDUAL_DATA_DIR, 'training')

# ============================================================================
# CANONICAL PATH PARAMETERS (Step 10)
# ============================================================================
# Number of canonical samples per chunk
K = 100  # Fixed number of points in [0,1] for canonical path

# Canonical signal: use spline values (normalized close from spline interpolation)
USE_SPLINE_VALUES = True  # If False, use normalized close from raw data

# ============================================================================
# DECODER MODEL PARAMETERS (Step 11)
# ============================================================================
DECODER_HIDDEN_LAYERS = [256, 128, 64]
DECODER_DROPOUT_RATES = [0.2, 0.2, 0.2]
DECODER_USE_BATCH_NORM = True
DECODER_LEARNING_RATE = 0.001
DECODER_WEIGHT_DECAY = 1e-5
DECODER_BATCH_SIZE = 512
DECODER_NUM_EPOCHS = 500
DECODER_EARLY_STOPPING_PATIENCE = 30
DECODER_EARLY_STOPPING_MIN_DELTA = 1e-6

# ============================================================================
# OHLCV RESIDUAL PARAMETERS (Step 12)
# ============================================================================
# Window size for canonical path context (days on each side)
CANONICAL_WINDOW_SIZE = 2  # p_{d-2}, p_{d-1}, p_d, p_{d+1}, p_{d+2}

# Include chunk-level features in per-day inputs
INCLUDE_CHUNK_FEATURES = False  # If True, append chunk feature vector to each day
NUM_CHUNK_FEATURES_TO_APPEND = 0  # If including, how many (0 = all)

# Volume residual target type
VOLUME_RESIDUAL_TYPE = 'log'  # 'log' for log(V), 'normalized' for standardized V

# ============================================================================
# RESIDUAL MODEL PARAMETERS (Step 13)
# ============================================================================
RESIDUAL_HIDDEN_LAYERS = [128, 64]
RESIDUAL_DROPOUT_RATES = [0.2, 0.2]
RESIDUAL_USE_BATCH_NORM = True
RESIDUAL_LEARNING_RATE = 0.001
RESIDUAL_WEIGHT_DECAY = 1e-5
RESIDUAL_BATCH_SIZE = 512
RESIDUAL_NUM_EPOCHS = 500
RESIDUAL_EARLY_STOPPING_PATIENCE = 30
RESIDUAL_EARLY_STOPPING_MIN_DELTA = 1e-6

# ============================================================================
# DATA SPLITTING
# ============================================================================
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# ============================================================================
# PATHS TO FIRST-STAGE DATA
# ============================================================================
# Spline data directory (from Step 6)
SPLINES_DIR = os.path.join(FIRST_STAGE_DATA_DIR, 'splines')

# Features directory (from Step 7)
FEATURES_DIR = os.path.join(FIRST_STAGE_DATA_DIR, 'features')

# Chunks directory (from Step 3)
CHUNKS_DIR = os.path.join(FIRST_STAGE_DATA_DIR, 'chunks')

# Normalized data directory (from Step 2) - for OHLCV residuals
NORMALIZED_DATA_DIR = os.path.join(FIRST_STAGE_DATA_DIR, 'normalized')

# Training sequences (from Step 8)
TRAINING_SEQUENCES_DIR = FIRST_STAGE_TRAINING_DIR

# ============================================================================
# HORIZON CALIBRATION (for T_future prediction)
# ============================================================================
# These are learned from calibration script (run once)
# T_future = HORIZON_A * geometric_shape_time_range + HORIZON_B
HORIZON_A = 1.0  # Will be updated by calibration script
HORIZON_B = 1.0  # Will be updated by calibration script
T_MIN = 2  # Minimum horizon (days)
T_MAX = 200  # Maximum horizon (days) - adjust based on your data

