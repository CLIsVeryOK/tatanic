from config.utils import AttrDict

config = AttrDict()
_C = config     # short alias to avoid coding

# task ----------------------
_C.TASK = AttrDict()
_C.TASK.NAME = ''
_C.TASK.MODE = 'train'
_C.TASK.OUTPUT_ROOT_DIR = '../results'
_C.TASK.GPUS = [0]


# data ----------------------
_C.DATA = AttrDict()
_C.DATA.ROOT_DIR = ''
_C.DATA.TRAIN_PATH = ''
_C.DATA.TRAIN_BATCH = 32

# solver ----------------------
_C.SOLVER = AttrDict()
_C.SOLVER.OPTIMIZER_NAME = 'SGD'
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.WEIGHT_DECAY = 0.002

_C.SOLVER.BASE_LR = 0.01
_C.SOLVER.FINETUNE = True
_C.SOLVER.FINETUNE_LR = 0.1
_C.SOLVER.LR_METHOD = 'step'
_C.SOLVER.LR_STEPS = [30, 50]
_C.SOLVER.LR_STEP_FACTOR = 0.1
_C.SOLVER.LR_DELAY_EPOCHS = 0.0
_C.SOLVER.LR_ETA_MIN = 0.00000077
_C.SOLVER.WARMUP_FACTOR = 1e-5
_C.SOLVER.WARMUP_EPOCH = 5

_C.SOLVER.APEX = False
_C.SOLVER.APEX_LEVEL = 'O1'

_C.SOLVER.START_EPOCH = 1
_C.SOLVER.EPOCHS = 60
_C.SOLVER.DISPLAY_INTERVAL = 100  # batch
_C.SOLVER.SAVE_INTERVAL = 2  # epoch
_C.SOLVER.START_SAVE = 100  # epoch
_C.SOLVER.RESUME_CHECKPOINT_IDX = 0  # epoch
_C.SOLVER.FT_MODEL_PATH = ''

