import argparse
import mxnet as mx
import numpy as np
from config.config import config, update_config
import os
import sys
import shutil
import pprint
curr_path = os.path.dirname(__file__)
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'lib'))
from utils.create_logger import create_logger
from symbols import *
def parse_args():
    parser = argparse.ArgumentParser(description='train fpn network')
    parser.add_argument('--cfg',help='configure file name',type = str, default = './cfgs/resnet_v1_101_coco_trainval_fpn_end2end_ohem.yaml')
    args, rest = parser.parse_known_args()
    update_config(args.cfg)

    return args

def train_net(args, ctx, pretrained, epoch, prefix, begin_epoch, end_epoch, lr, lr_step):
    mx.random.seed(3)
    np.random.seed(3)
    logger, final_output_path = create_logger(config.output_path, args.cfg, config.dataset.image_set)
    prefix = os.path.join(final_output_path, prefix)

    shutil.copy2(os.path.join(curr_path, 'symbols',config.symbol+'.py'),final_output_path)
    sym_instance = eval(config.symbol + '.' + config.symbol)()
    sym = sym_instance.get_symbol(config, is_train = True)

    feat_pyramid_level = np.log2(config.network.RPN_FEAT_STRIDE).astype(int)
    feat_sym = [sym.get_internals()['rpn_cls_score_p'+str(x)+'_output'] for x in feat_pyramid_level]

    batch_size = len(ctx)
    input_batch_size = config.TRAIN.BATCH_IMAGES * batch_size

    pprint.pprint(config)
    logger.info('training config:{}\n'.format(pprint.pformat(config)))

    image_sets = [iset for iset in config.dataset.image_set.split('+')]

def main():
    args = parse_args()
    print('called with argument:',args)
    print(args)
    ctx = [mx.gpu(int(i)) for i in config.gpus.split(',')]
    train_net(args, ctx, config.network.pretrained, config.network.pretrained_epoch, 
              config.TRAIN.model_prefix,config.TRAIN.begin_epoch, config.TRAIN.end_epoch,
              config.TRAIN.lr, config.TRAIN.lr_step)
if __name__ == '__main__':
    main()