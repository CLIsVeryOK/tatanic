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
