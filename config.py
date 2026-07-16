import torch

#------------------数据相关-----------------------
DATA_PATH = './data'
BATCH_SIZE = 128
VALID_SPLIT = 0.1
NUM_WORKERS = 2

#------------------模型结构------------------------
INPUT_CHANNEL = 1
CONV1_OUT = 32
CONV2_OUT = 64
KERNEL_SIZE = 3
POOL_KERNEL = 2
FC1_UNITS = 128
FC2_UNITS = 10
NUM_CLASS = 10
DROPOUT = 0.25

#------------------训练超参数----------------------
EPOCHES = 10
LEARNING_RATE = 0.001
MOMENTUM = 0.9
WEIGHT_DCAY = 1e-5

#------------------设备---------------------------
LOG_INTERVAL = 100
SAVE_BEST = True
MODEL_SAVE_PATH = "./best_model.pth"

#-----------------随机种子------------------------
SEED = 42