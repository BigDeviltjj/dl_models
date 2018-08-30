import tensorflow as tf

from datasets import pascalvoc_to_tfrecords

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
        'dataset_dir','/home/miracle/tf/faster_rcnn/tf-faster-rcnn/data/VOCdevkit2007/VOC2007/','directory of VOC dataset')
tf.app.flags.DEFINE_string(
        'output_dir','./tfrecords',
        'output dir of tfrecords files')
tf.app.flags.DEFINE_string(
        'output_name','VOC2007_tfrecord',
        'output name of tfrecords files')


def main(_):
    print('dataset dir:{}'.format(FLAGS.dataset_dir))
    print('output dir:{}'.format(FLAGS.output_dir))
    pascalvoc_to_tfrecords.run(FLAGS.dataset_dir,FLAGS.output_dir, FLAGS.output_name)

if __name__ == '__main__':
    tf.app.run()

 

