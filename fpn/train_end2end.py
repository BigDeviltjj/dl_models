import argparse
import mxnet as mx
import numpy as np
import os
import sys
import shutil
import pprint
from config.config import config, update_config
curr_path = os.path.dirname(__file__)
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'lib'))
from utils.create_logger import create_logger
from utils.load_data import load_gt_roidb, merge_roidb, filter_roidb
from utils.load_model import load_param
from symbols import resnet_v1_101_fpn_rcnn
from core.loader import PyramidAnchorIterator

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
    sym_instance = resnet_v1_101_fpn_rcnn.resnet_v1_101_fpn_rcnn()
    sym = sym_instance.get_symbol(config, is_train = True)

    feat_pyramid_level = np.log2(config.network.RPN_FEAT_STRIDE).astype(int)
    feat_sym = [sym.get_internals()['rpn_cls_score_p'+str(x)+'_output'] for x in feat_pyramid_level]

    batch_size = len(ctx)
    input_batch_size = config.TRAIN.BATCH_IMAGES * batch_size

    pprint.pprint(config)
    logger.info('training config:{}\n'.format(pprint.pformat(config)))

    image_sets = [iset for iset in config.dataset.image_set.split('+')]
    roidbs = [load_gt_roidb(config.dataset.dataset,image_set, config.dataset.root_path, config.dataset.dataset_path,
                            flip = config.TRAIN.FLIP) for image_set in image_sets]
        
    roidb = merge_roidb(roidbs)
    roidb = filter_roidb(roidb,config)
    
    train_data = PyramidAnchorIterator(feat_sym, roidb, config,batch_size = input_batch_size, shuffle = config.TRAIN.SHUFFLE,
                                   ctx = ctx, feat_strides = config.network.RPN_FEAT_STRIDE, anchor_scales = config.network.ANCHOR_SCALES,
                                   anchor_ratios = config.network.ANCHOR_RATIOS, aspect_grouping = config.TRAIN.ASPECT_GROUPING,
                                   allowed_border = np.inf)

    max_data_shape = [('data',(config.TRAIN.BATCH_IMAGES,3,max([v[0] for v in config.SCALES]),max([v[1] for v in config.SCALES])))]
    max_data_shape,max_label_shape = train_data.infer_shape(max_data_shape)
    max_data_shape.append(('gt_boxes',(config.TRAIN.BATCH_IMAGES,100,5)))
    print('providing maximum shape', max_data_shape, max_label_shape)

    data_shape_dict = dict(train_data.provide_data_single + train_data.provide_label_single)
    pprint.pprint(data_shape_dict)
    sym_instance.infer_shape(data_shape_dict)

    if config.TRAIN.RESUME:
        print('continue training from ',begin_epoch)
        arg_params, aux_params = load_param(prefix, begin_epoch, convert = True)
    else:
        arg_params, aux_params = load_param(pretrained, epoch, convert = True)
        sym_instance.init_weight(config, arg_params, aux_params)

    sym_instance.check_parameter_shapes(arg_params, aux_params, data_shape_dict)
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