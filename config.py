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
CONV3_OUT = 128
KERNEL_SIZE = 3
POOL_KERNEL = 2
FC1_UNITS = 128
FC2_UNITS = 64
NUM_CLASS = 10
DROPOUT = 0.5

#------------------训练超参数----------------------
EPOCHES = 30
LEARNING_RATE = 0.001
MOMENTUM = 0.9
WEIGHT_DCAY = 5e-4

#------------------设备---------------------------
LOG_INTERVAL = 100
SAVE_BEST = True
MODEL_SAVE_PATH = "./best_model.pth"

#-----------------随机种子------------------------
SEED = 42

#----------------数据增强-------------------------
AUG_ROTATION_DEGREES = 15       # 随机旋转角度范围（±度数）
AUG_TRANSLATE_RATIO = 0.02     # 随机平移比例（相对于图像尺寸）

#----------------GPU相关-------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')