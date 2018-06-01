from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import tarfile

from six.moves import urllib
import tensorflow as tf

DATA_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'


parser = argparse.ArgumentParser()

parser.add_argument(
    '--data-dir',type = str,default = '/home/miracle/dl_models/cifar10',
    help = "Directory to download data and extract the tarball"
)
parser.add_argument(
    '--data-dir',type = str,default = '/home/miracle/dl_models/cifar10',
    help = "Directory to download data and extract the tarball"
)

def main(_):
    if not os.path.exists(FLAGS.data_dir):
        os.makedirs(FLAGS.data_dir)
    
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(FLAGS.data_dir,filename)

    if not os.path.exists(filepath):
        def _progress(count,block_size,total_size):
            sys.stdout.write('\r>>Downloading %s %.1f%%'%(
                filename,100.0*count*block_size/total_size
            ))
        filepath, _ = urllib.request.urlretrieve(DATA_URL,filepath,_progress)

        print()
        stateinfo=os.stat(filepath)
        print('Successfully downloaded',filename,stateinfo.st_size,'bytes.')

    tarfile.open(filepath,'r:gz').extractall(FLAGS.data_dir)

if __name__=='__main__':
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(argv=[sys.argv[0]]+unparsed)