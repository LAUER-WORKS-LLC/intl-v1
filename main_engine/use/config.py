"""
Configuration for Inference Pipeline
"""

import os

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Model paths
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')

# Paths to training data (for scalers and feature extraction)
FIRST_STAGE_DATA_DIR = os.path.join(BASE_DIR, '00_MODELING', 'a_feature_modeling', '09_train_features', 'data')
FIRST_STAGE_TRAINING_DIR = os.path.join(FIRST_STAGE_DATA_DIR, 'training')
CHUNKS_DIR = os.path.join(FIRST_STAGE_DATA_DIR, 'chunks')

# Model file names (best models)
POWERHOUSE_MODEL = 'feature_predictor_powerhouse_best_20251125_042420'
CANONICAL_DECODER_MODEL = 'canonical_path_decoder_best_20251125_153921'
RESIDUAL_MODEL = None  # Will be set when Step 13 completes

# Horizon calibration parameters
HORIZON_A = 1.0  # Will be updated by calibration script
HORIZON_B = 1.0  # Will be updated by calibration script
T_MIN = 2
T_MAX = 200

# Canonical path parameters
K = 100  # Number of canonical points
CANONICAL_WINDOW_SIZE = 2  # For residual model input

# Number of previous chunks for input
NUM_PREVIOUS_CHUNKS = 3

