import tensorflow as tf
import numpy as np
import os 

VGG_MEAN = [103.939, 116.779, 123.68]

class Vgg16:
    def __init__(self,path = None):
        #path = os.path.abspath('vgg16.npy')
        self.data_dict = np.load(path,encoding='latin1').item()
        self.lr = 1e-2
        self.global_step = tf.Variable(tf.constant(0),trainable=False,name='global_step')
    def _import_data(self):
        with tf.name_scope('data'):
            self.data_X = tf.placeholder(tf.float32,shape=[None,224,224,3],name='data_X')
            self.labels = tf.placeholder(tf.float32,shape=[None],name='labels')

    def _conv_layer(self,input,name):
        with tf.variable_scope(name):
            self.filter = tf.Variable(self.data_dict[name][0],name=name+'_filter')
            biases = tf.Variable(self.data_dict[name][1],name=name+'_biases')
            x = tf.nn.conv2d(input,self.filter,[1,1,1,1],padding='SAME') + biases
        return tf.nn.relu(x)
    
    def _max_pool(self,input,name):
        with tf.name_scope(name):
            x = tf.nn.max_pool(input,[1,2,2,1],[1,2,2,1],padding = "SAME")
        return x
    
    def _fc_layer(self,input,name):
        with tf.variable_scope(name):
            b = tf.Variable(self.data_dict[name][1],name=name+'_biases')
            if name == 'fc6':
                w = tf.Variable(self.data_dict[name][0].reshape((7,7,512,4096)),name=name+'_weights')
            elif name == 'fc7':
                w = tf.Variable(self.data_dict[name][0].reshape((1,1,4096,4096)),name=name+'_weights')
            else:
                w = tf.Variable(self.data_dict[name][0].reshape((1,1,4096,1000)),name=name+'_weights')
            x = tf.nn.conv2d(input,w,[1,1,1,1],padding="VALID")+b
        return x
    def _build_vgg16(self,is_training):
        self.is_training = is_training
        rgb_scaled = self.data_X * 255.0

        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
        
        conv1_1 = self._conv_layer(bgr,'conv1_1')
        conv1_2 = self._conv_layer(conv1_1,'conv1_2')
        max_pool1 = self._max_pool(conv1_2,'max_pool1')
        conv2_1 = self._conv_layer(max_pool1,'conv2_1')
        conv2_2 = self._conv_layer(conv2_1,'conv2_2')
        max_pool2 = self._max_pool(conv2_2,'max_pool2')

        conv3_1 = self._conv_layer(max_pool2,'conv3_1')
        conv3_2 = self._conv_layer(conv3_1,'conv3_2')
        conv3_3 = self._conv_layer(conv3_2,'conv3_3')
        max_pool3 = self._max_pool(conv3_3,'max_pool3')

        conv4_1 = self._conv_layer(max_pool3,'conv4_1')
        conv4_2 = self._conv_layer(conv4_1,'conv4_2')
        conv4_3 = self._conv_layer(conv4_2,'conv4_3')
        max_pool4 = self._max_pool(conv4_3,'max_pool4')

        conv5_1 = self._conv_layer(max_pool4,'conv5_1')
        conv5_2 = self._conv_layer(conv5_1,'conv5_2')
        conv5_3 = self._conv_layer(conv5_2,'conv5_3')
        max_pool5 = self._max_pool(conv5_3,'max_pool5')
        fc6 = self._fc_layer(max_pool5,'fc6')
        fc6 = tf.nn.relu(fc6)
        if self.is_training is True:
            fc6 = tf.nn.dropout(fc6,0.5)
            pass
        fc7 = self._fc_layer(fc6,'fc7')
        fc7 = tf.nn.relu(fc7)
        if self.is_training is True:
            fc7 = tf.nn.dropout(fc7,0.5)

        self.logits = self._fc_layer(fc7,'fc8')
        self.logits = tf.reduce_mean(self.logits,axis = [1,2],name='logits')
    def _create_loss(self):
        with tf.name_scope('loss'):
            self.loss = tf.losses.softmax_cross_entropy(onehot_labels=tf.one_hot(tf.cast(self.labels,tf.int32),1000),logits = self.logits)

    def _create_optimizer(self):
        with tf.name_scope('optimizer'):
            self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss,global_step=self.global_step)
    def _create_summaries(self):
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss',self.loss)
            tf.summary.histogram('his_loss',self.loss)
            self.summary_op = tf.summary.merge_all()
    def build_graph(self,is_training =  False):
        self._import_data()
        self._build_vgg16(is_training)
        self._create_loss()
        self._create_optimizer()
        self._create_summaries()