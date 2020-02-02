from Utilities import * 
import os

"""
Hyperparameters for various 
parameters like how to construct 
trees, network learning_rate, decays,
whether to use GPU or not etc
"""
# Dataset directory
DATA_DIRECTORY = './Emojis/Train/'

# Vector Dimensions
PATH_CODE_SIZE = 24
FEATURE_SIZE = 80
HIDDEN_SIZE = 200

# Logging during training
SHOW_LOG_EVERY = 3
SAVE_LOG = True
SAVE_LOG_EVERY = 3
SAVE_SNAPSHOT = True
SAVE_SNAPSHOT_EVERY = 5

# Training Hyperparameters
EPOCHS = 100
BATCH_SIZE = 500

LR = .001
LR_DECAY_BY = 1
LR_DECAY_EVERY = 1

CUDA = True
GPU = 0

# Descriptor functions
DESC_FUNCTIONS = [
    fd,             # 20 dim vector
    relbb           # 4 dim vector
]

# Path relation functions
RELATION_FUNCTIONS = [
    symGraph,
    adjGraph
] 

# Graph clustering functions
GRAPH_CLUSTER_ALGO = {
    "KL" : community.kernighan_lin_bisection,
}

# Where to save stuff
MODEL_SAVE_PATH = './Models/'
RESUME_SNAPSHOT = ''

SAVE_TREES = False
SAVE_TREES_DIR = './Trees/Train/'
