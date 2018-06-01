from tensorflow import flags
import tensorflow as tf

def set_defaults(**kwargs):
    for key,value in kwargs.items():
        flags.FLAGS.set_default(name=key,value = value)


set_defaults(data_dir='/tmp/cifar10_data')
flags.DEFINE_string(
        name="data_dir", short_name="dd", default="/tmp",
        help='lalala')
print(flags.FLAGS)