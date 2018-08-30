import tensorflow as tf

import os
import tensorflow as tf
import os
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
VOC_ID_NAME = {}
for k,v in VOC_LABELS.items():
    VOC_ID_NAME[v[0]] = k
def _parse_func(example):
    	keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/height': tf.FixedLenFeature([1], tf.int64),
        'image/width': tf.FixedLenFeature([1], tf.int64),
        'image/channels': tf.FixedLenFeature([1], tf.int64),
        'image/shape': tf.FixedLenFeature([3], tf.int64),
        'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/label': tf.VarLenFeature(dtype=tf.int64),
        'image/object/bbox/difficult': tf.VarLenFeature(dtype=tf.int64),
        'image/object/bbox/truncated': tf.VarLenFeature(dtype=tf.int64)}
	parsed_features = tf.parse_single_example(example, keys_to_features)
	img = parsed_features['image/encoded']
	img = tf.image.decode_jpeg(img)
	labels = parsed_features['image/object/bbox/label']
	xmin = parsed_features['image/object/bbox/xmin']
	ymin = parsed_features['image/object/bbox/ymin']
	xmax = parsed_features['image/object/bbox/xmax']
	ymax = parsed_features['image/object/bbox/ymax']
	shape = parsed_features['image/shape']
	return img,(labels,xmin,ymin,xmax,ymax),shape

filenames = os.listdir('./')
dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.map(_parse_func)
iterator = dataset.make_one_shot_iterator()

next_img = iterator.get_next()

import cv2
with tf.Session() as sess:
    for i in range(1000):
        img,bbox,shape = sess.run(next_img)
        labels = bbox[0][1]
        xmin = bbox[1][1]
        ymin = bbox[2][1]
        xmax = bbox[3][1]
        ymax = bbox[4][1]
        img = img[:,:,::-1]
        for j in range(len(labels)):
            b,g,r = cv2.split(img)
            img = cv2.merge([b,g,r])
            xmin_cood = int(xmin[j] * shape[1])
            ymin_cood = int(ymin[j] * shape[0])
            xmax_cood = int(xmax[j] * shape[1])
            ymax_cood = int(ymax[j] * shape[0])
            cv2.putText(img,VOC_ID_NAME[labels[j]],(xmin_cood,ymin_cood),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),3)
            tl = (xmin_cood,ymin_cood)
            br = (xmax_cood,ymax_cood)
            cv2.rectangle(img,tl,br,(0,0,255),2)

            #cv2.rectangle(img,(100,150),(110,160),(255,0,0),2)
            cv2.imshow('img',img)
            cv2.waitKey()
