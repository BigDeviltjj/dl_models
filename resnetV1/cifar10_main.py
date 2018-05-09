from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow import app as tf_app
from tensorflow import flags
import argparse
import resnet_model
import resnet_run_loop
import os
_HEIGHT = 32
_WIDTH = 32
_NUM_CHANNELS = 3
_DEFAULT_IMAGE_BYTES = _HEIGHT * _WIDTH * _NUM_CHANNELS
# The record is the image plus a one-byte label
_RECORD_BYTES = _DEFAULT_IMAGE_BYTES + 1
_NUM_CLASSES = 10
_NUM_DATA_FILES = 5

_NUM_IMAGES = {
    'train': 50000,
    'validation': 10000,
}

parser = argparse.ArgumentParser()

parser.add_argument(
    '--data-dir',type = str,default = '/home/miracle/dl_models/cifar10',
    help = "Directory to download data and extract the tarball"
)
parser.add_argument(
    '--model_dir',type = str,default = '/home/miracle/dl_models/pretrained_models/resnet_cifar10',
    help = "model_dir"
)
parser.add_argument(
    '--resnet_size',type = str,default = '32',
    help = "resnet_size"
)
parser.add_argument(
    '--train_epochs',type = int,default = 250,
    help = "train_epochs"
)
parser.add_argument(
    '--epochs_between_evals',type = int,default = 10,
    help = "epochs_between_evals"
)
parser.add_argument(
    '--batch_size',type = int,default = 128,
    help = "batch_size"
)
parser.add_argument(
    '--data_format',type = str,default = None,
    help = "batch_size"
)


DATASET_NAME = 'CIFAR-10'
def define_cifar_flags():
    FLAGS,_ =  parser.parse_known_args()
    return FLAGS

def get_filenames(is_training,data_dir):
    data_dir = os.path.join(data_dir,'cifar-10-batches-bin')
    assert os.path.exists(data_dir), (
      'Run cifar10_download_and_extract.py first to download and extract the '
      'CIFAR-10 data.')
    if is_training:
        return [os.path.join(data_dir,'data_batch_%d.bin'%i)
            for i in range (1,_NUM_DATA_FILES+1)]
    else:
        return [os.path.join(data_dir,'test_batch.bin')]

def preprocess_image(image,is_training):
    if is_training:
        image = tf.image.resize_image_with_crop_or_pad(image,_HEIGHT+8,_WIDTH+8)
        image = tf.random_crop(image,[_HEIGHT,_WIDTH,_NUM_CHANNELS])
        image = tf.image.random_flip_left_right(image)
    
    image = tf.image.per_image_standardization(image)
    return image
def parse_record(raw_record,is_training):
    record_vector = tf.decode_raw(raw_record,tf.uint8)
    label = tf.cast(record_vector[0],tf.int32)
    label = tf.one_hot(label,_NUM_CLASSES)

    depth_major = tf.reshape(record_vector[1:_RECORD_BYTES],[_NUM_CHANNELS,_HEIGHT,_WIDTH])
    image = tf.cast(tf.transpose(depth_major,[1,2,0]),tf.float32)
    image = preprocess_image(image,is_training)
    return image, label

def input_fn(is_training,data_dir,batch_size,num_epochs=1):
    filenames = get_filenames(is_training,data_dir)
    dataset = tf.data.FixedLengthRecordDataset(filenames,_RECORD_BYTES)

    return resnet_run_loop.process_record_dataset(dataset,is_training,batch_size,_NUM_IMAGES['train'],
                        parse_record,num_epochs)
class Cifar10Model(resnet_model.Model):
    def __init__(self,resnet_size,data_format=None,num_classes=_NUM_CLASSES,
                 resnet_version=resnet_model.DEFAULT_VERSION,
                 dtype = resnet_model.DEFAULT_DTYPE):
        num_blocks = (resnet_size - 2) // 6
        super(Cifar10Model,self).__init__(
            resnet_size=resnet_size,
            bottleneck=False,
            num_classes=num_classes,
            num_filters=16,
            kernel_size=3,
            conv_stride=1,
            first_pool_size=None,
            first_pool_stride=None,
            block_sizes=[num_blocks] * 3,
            block_strides=[1, 2, 2],
            final_size=64,
            resnet_version=resnet_version,
            data_format=data_format,
            dtype=dtype           
        )
def cifar_10_model_fn(features,labels,mode,params):
    print(features)
    features = tf.reshape(features,[-1,_HEIGHT,_WIDTH,_NUM_CHANNELS])
    learning_rate_fn = resnet_run_loop.learning_rate_with_decay(batch_size=params['batch_size'],
                    batch_denom=128,num_images=_NUM_IMAGES['train'],boundary_epochs=[100,150,200],
                    decay_rates=[1,0.1,0.01,0.001])
    weight_decay =  2e-4
    def loss_filter_fn(_):
        return True
    
    return resnet_run_loop.resnet_model_fn(
        features=features,
        labels = labels,
        mode = mode,
        model_class = Cifar10Model,
        resnet_size=params['resnet_size'],
        weight_decay=weight_decay,
        learning_rate_fn=learning_rate_fn,
        momentum=0.9,
        data_format=params['data_format'],
        resnet_version=params['resnet_version'],
        loss_scale = params['loss_scale'],
        loss_filter_fn=loss_filter_fn,
        dtype=params['dtype']
    )
def run_cifar(flags_obj):
    input_function = input_fn

    resnet_run_loop.resnet_main(
        flags_obj,cifar_10_model_fn,input_function,DATASET_NAME,
        shape = [_HEIGHT,_WIDTH,_NUM_CLASSES]
    )
def main(_):
    flags_obj = define_cifar_flags()
    run_cifar(flags_obj)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf_app.run()