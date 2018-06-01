from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from configure import cfg
from model import Densenet
import tensorflow as tf
import argparse
import matplotlib.pyplot as plt
import numpy as np
def parse_args():

    parser = argparse.ArgumentParser(description="Train/test a densenet")
    parser.add_argument('--mode',dest="mode",type = str,default = 'TRAIN',
                        help = "mode,train,test or eval")
    parser.add_argument('--L',dest="L",type = int,default = 40,
                        help = "depth of each block")
    parser.add_argument('--k',dest="k",type = int,default = 12,
                        help = "growth rate")
    parser.add_argument('--N',dest="N",type = int,default = 3,
                        help = "number of blocks")
    parser.add_argument('--theta',dest="theta",type = int,default = 1,
                        help = "compress factor")
    parser.add_argument('--dataset',dest="dataset",type = str,default = "cifar10",
                        help = "dataset,cifar10,cifar100 or imagenet")
    parser.add_argument('--datapath',dest="datapath",type = str,default = "/home/miracle/dl_models/cifar10",
                        help = "path of data")
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()
    densenet = Densenet(args)
    densenet.build_net()
    #densenet.train()
    densenet.train()
    plt.show()

 