from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from easydict import EasyDict as edict

_C = edict()

cfg = _C

_C.TRAIN = edict()
_C.TRAIN.batch_size = 64
_C.TRAIN.KEEP_PROB = 0.8
_C.TRAIN.epochs = 300
_C.TRAIN.VERBOSE = 20
_C.TRAIN.boundary_epochs = [0.5,0.75]
_C.TRAIN.decay_rates = [1,0.1,0.01]


_C.NUM_DATA_FILES = 5
_C.LABEL_BYTES = 1
_C.HEIGHT = 32
_C.WIDTH = 32
_C.CHANNELS = 3
_C.RECORD_BYTES = 32*32*3 + 1
_C.NUM_DATA_FILES_TRAIN = 50000
_C.NUM_DATA_FILES_VAL = 10000
_C.NUM_CLASSES = 10
_C.MEAN_IMAGE = 128
_C.LR = 0.1
_C.MOMENTUM = 0.9

_C.WEIGHT_DECAY = 1e-4