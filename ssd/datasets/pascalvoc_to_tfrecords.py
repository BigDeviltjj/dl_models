import tensorflow as tf
import os
import sys
import random
import xml.etree.ElementTree as ET

def int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def float_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def bytes_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


VOC_LABELS = {
    'none': (0, 'Background'),
    'aeroplane': (1, 'Vehicle'),
    'bicycle': (2, 'Vehicle'),
    'bird': (3, 'Animal'),
    'boat': (4, 'Vehicle'),
    'bottle': (5, 'Indoor'),
    'bus': (6, 'Vehicle'),
    'car': (7, 'Vehicle'),
    'cat': (8, 'Animal'),
    'chair': (9, 'Indoor'),
    'cow': (10, 'Animal'),
    'diningtable': (11, 'Indoor'),
    'dog': (12, 'Animal'),
    'horse': (13, 'Animal'),
    'motorbike': (14, 'Vehicle'),
    'person': (15, 'Person'),
    'pottedplant': (16, 'Indoor'),
    'sheep': (17, 'Animal'),
    'sofa': (18, 'Indoor'),
    'train': (19, 'Vehicle'),
    'tvmonitor': (20, 'Indoor'),
}


def _add_to_tfrecord(dataset, img_name, tfrecord_writer):
    image_path = dataset + 'JPEGImages/' + img_name + '.jpg'
    image_data = tf.gfile.FastGFile(image_path,'r').read()
    
    filename = os.path.join(dataset,'Annotations/',img_name + '.xml')
    tree = ET.parse(filename)
    root = tree.getroot()

    size  = root.find('size')
    shape = [int(size.find('height').text),
             int(size.find('width').text),
             int(size.find('depth').text)]

    bboxes = []
    labels = []
    labels_text = []
    difficult = []
    truncated = []
    for obj in root.findall('object'):
        label = obj.find('name').text
        labels.append(int(VOC_LABELS[label][0]))
        labels_text.append(label.encode('ascii'))

        if obj.find('difficult'):
            difficult.append(int(obj.find('difficult').text))
        else:
            difficult.append(0)
        if obj.find('truncated'):
            truncated.append(int(obj.find('truncated').text))
        else:
            truncated.append(0)

        bbox = obj.find('bndbox')
        bboxes.append((float(bbox.find('ymin').text)/shape[0],
                       float(bbox.find('xmin').text)/shape[1],
                       float(bbox.find('ymax').text)/shape[0],
                       float(bbox.find('xmax').text)/shape[1]))

    xmin = []
    ymin = []
    xmax = []
    ymax = []

    for b in bboxes:
        [l.append(point) for l, point in zip([ymin,xmin,ymax, xmax],b)]
    
    image_format = b'JPEG'
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': int64_feature(shape[0]),
        'image/width': int64_feature(shape[1]),
        'image/channels': int64_feature(shape[2]),
        'image/shape': int64_feature(shape),
        'image/object/bbox/xmin': float_feature(xmin),
        'image/object/bbox/ymin': float_feature(ymin),
        'image/object/bbox/xmax': float_feature(xmax),
        'image/object/bbox/ymax': float_feature(ymax),
        'image/object/bbox/label': int64_feature(labels),
        'image/object/bbox/label_text': bytes_feature(labels_text),
        'image/object/bbox/difficult': int64_feature(difficult),
        'image/object/bbox/truncated': int64_feature(truncated),
        'image/format': bytes_feature(image_format),
        'image/encoded': bytes_feature(image_data)}))
    tfrecord_writer.write(example.SerializeToString())

def run(dataset_dir, output_dir, name = 'voc_train', shuffling = False):
    if not tf.gfile.Exists(output_dir):
        tf.gfile.MakeDirs(output_dir)
    
    path = os.path.join(dataset_dir,'Annotations/')
    filenames = sorted(os.listdir(path))
    if shuffling:
        random.seed(4242)
        random.shuffle(filenames)
    
    i = 0
    fidx = 0
    while i < len(filenames):
        tf_filename = '{}/{}_{}.tfrecord'.format(output_dir,name,fidx)
        with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
            j = 0
            while i < len(filenames) and j < 200:
                sys.stdout.write('\r>>Converting image {}/{}'.format(i,len(filenames)))
                sys.stdout.flush()

                filename = filenames[i]
                img = filename[:-4]
                _add_to_tfrecord(dataset_dir,img,tfrecord_writer)
                i += 1
                j += 1
            fidx += 1
    print('\nFinished converting pascalvoc dataset')
