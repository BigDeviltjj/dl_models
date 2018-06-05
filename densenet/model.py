from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from configure import cfg
import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
def parse_record_fn(value,is_training):
    record_vector = tf.decode_raw(value,tf.uint8)
    labels = tf.cast(record_vector[0],tf.int32)
    labels = tf.one_hot(labels,cfg.NUM_CLASSES)

    image = tf.reshape(record_vector[1:cfg.RECORD_BYTES],[cfg.CHANNELS,cfg.HEIGHT,cfg.WIDTH])
    image = tf.cast(tf.transpose(image,[1,2,0]),tf.float32)
    
    if is_training:
        image = tf.image.resize_image_with_crop_or_pad(image,cfg.HEIGHT+8,cfg.WIDTH+8)
        image = tf.random_crop(image,[cfg.HEIGHT,cfg.WIDTH,cfg.CHANNELS])
        image = tf.image.random_flip_left_right(image)
    #image = tf.image.per_image_standardization(image)
    image = image/128. - 1
    return image,labels

def get_filenames(is_training,data_dir):
    data_dir = os.path.join(data_dir,'cifar-10-batches-bin')
    if is_training:
        return [os.path.join(data_dir,'data_batch_%d.bin'%i)
                for i in range(1,cfg.NUM_DATA_FILES+1)]
    else:
        return [os.path.join(data_dir,'test_batch.bin')]


def conv_layer(inputs,filters,k_size,stride,padding,is_training,scope_name='conv',use_bias=False):
    with tf.variable_scope(scope_name) as scope:
        in_channels = inputs.shape[-1]
        std = tf.sqrt(2./(k_size*k_size*int(in_channels)))
        kernel = tf.get_variable('weights',[k_size,k_size,in_channels,filters],
                                 initializer=tf.random_normal_initializer(stddev=std))    #std=sqrt(2/(k*k*in_channels))
        conv = tf.nn.conv2d(inputs,kernel,[1,stride,stride,1],padding=padding)
        if use_bias is True:
            biases = tf.get_variable('biases',[filters],
                                     initializer=tf.zeros_initializer())
            conv += biases
        #if is_training==True:
            #conv = tf.nn.dropout(conv, cfg.TRAIN.KEEP_PROB, name='relu_dropout')
    return conv

def relu_layer(inputs,scope_name = 'relu'):
    with tf.name_scope(scope_name):
        x = tf.nn.relu(inputs)
    return x
def bn_layer(inputs,is_training,scope_name='bn'):
    with tf.name_scope(scope_name):
        x = tf.layers.batch_normalization(inputs=inputs,training = is_training)  
    return x
def avg_pool(inputs,k_size,stride,padding,scope_name = 'avg_pool'):
    with tf.name_scope(scope_name):
        x = tf.nn.avg_pool(inputs,[1,k_size,k_size,1],strides=[1,stride,stride,1],padding=padding)
    return x
def fc_layer(inputs,out_dims,is_training,scope_name='fc'):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        in_dim = inputs.shape[-1]
        w = tf.get_variable('weights', [in_dim, out_dims],
                            initializer=tf.truncated_normal_initializer())
        b = tf.get_variable('biases', [out_dims],
                            initializer=tf.constant_initializer(0.0))
        out = tf.matmul(inputs, w) + b
    return out

class Densenet():
    def __init__(self,args):
        self.L = args.L
        self.k = args.k
        self.N =  int((self.L-4)/3)
        self.theta = args.theta
        self.dataset = args.dataset
        self.datapath = args.datapath
        self.is_training = tf.placeholder(tf.bool)
        self.feature_map={}
        self.gstep = tf.Variable(0, dtype=tf.int32, 
                                trainable=False, name='global_step')
        self.initial_lr = cfg.LR
        print("init finished\n")


    def __process_record_dataset(self,dataset,is_training,batch_size,num_data_files,num_epochs):
        #dataset = dataset.prefetch(buffer_size=batch_size)
        if is_training:
            dataset =dataset.shuffle(buffer_size=num_data_files)
        #dataset = dataset.repeat(num_epochs)
        dataset = dataset.map(lambda value:parse_record_fn(value,is_training)).batch(batch_size)
        return dataset

    def __input_func(self,is_training,data_dir,batch_size,num_epochs):
        filenames = get_filenames(is_training,data_dir)
        dataset = tf.data.FixedLengthRecordDataset(filenames,cfg.RECORD_BYTES)

        return self.__process_record_dataset(dataset,is_training,batch_size,cfg.NUM_DATA_FILES_TRAIN,num_epochs)
    def __build_input(self):
        if self.dataset == "cifar10":
            # self.images,self.labels = tf.cond(self.is_training,lambda:self.__input_func(is_training=True,data_dir=self.datapath,
            #             batch_size=cfg.TRAIN.batch_size,
            #             num_epochs = cfg.TRAIN.epochs),lambda:self.__input_func(is_training=False,data_dir=self.datapath,
            #             batch_size=cfg.TRAIN.batch_size,
            #             num_epochs = 1))


            self.train_data = self.__input_func(is_training=True,data_dir=self.datapath,
                         batch_size=cfg.TRAIN.batch_size,
                         num_epochs = cfg.TRAIN.epochs)
            self.test_data = self.__input_func(is_training=False,data_dir=self.datapath,
                         batch_size=cfg.TRAIN.batch_size,
                         num_epochs = 1)
            iterator = tf.data.Iterator.from_structure(self.train_data.output_types, 
                                                   self.train_data.output_shapes)
            # 
            self.images,self.labels = iterator.get_next()    
            self.train_init = iterator.make_initializer(self.train_data)
            self.test_init = iterator.make_initializer(self.test_data)


    def __add_dense_block(self,inputs,i):
        x = inputs
        with tf.variable_scope('dense_%d'%i) as scope:
            for index in range(self.N):
                out = bn_layer(x,is_training=self.is_training)
                out = relu_layer(out)
                out = conv_layer(out,self.k,3,1,'SAME',self.is_training,scope_name='conv_%d'%index)
                x = tf.concat([out,x],axis = -1)
        return x
    def __add_transition_layer(self,inputs,i):
        x = inputs
        with tf.variable_scope('transition_%d'%i) as scope:
            x = bn_layer(x,is_training= self.is_training)
            x = relu_layer(x)
            filters = int(self.theta * int(x.shape[-1]))
            x = conv_layer(x,filters,1,1,'SAME',is_training = self.is_training)
            x = avg_pool(x,2,2,'VALID')
        return x
    def __build_network(self):
        x = self.images
        x = conv_layer(x,16,3,1,'SAME',self.is_training,'conv_1')
        for i in range(3):
            x = self.__add_dense_block(x,i+1)
            self.feature_map['dense_blok%d'%(i+1)] = x
            if i is not 2:
                x = self.__add_transition_layer(x,i+1)
            else:
                x = relu_layer(bn_layer(x,self.is_training))
        x = tf.reduce_mean(x,axis = [1,2],keepdims = False)
        self.logits = fc_layer(x,10,self.is_training,'fc')


    def __build_loss(self):
        with tf.name_scope('loss'):
            self.l2_loss = cfg.WEIGHT_DECAY * tf.add_n(
                            [tf.nn.l2_loss(tf.cast(v,tf.float32)) 
                             for v in tf.trainable_variables()
                             if 'weights' in v.name])
            entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.labels,logits=self.logits)
            self.loss = tf.reduce_mean(entropy) + self.l2_loss 


    def __build_optimizer(self):
        with tf.name_scope('optimizer'):
            boundaries = [int(cfg.NUM_DATA_FILES_TRAIN/cfg.TRAIN.batch_size * cfg.TRAIN.epochs * epoch) 
                            for epoch in cfg.TRAIN.boundary_epochs]
            vals = [self.initial_lr * decay for decay in cfg.TRAIN.decay_rates]
            self.lr = tf.train.piecewise_constant(self.gstep,boundaries,vals)
            optimizer = tf.train.MomentumOptimizer(self.lr,momentum=cfg.MOMENTUM,use_nesterov=True)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.opt = optimizer.minimize(self.loss,global_step=self.gstep)


    def __build_eval(self):
        with tf.name_scope('predict'):
            preds = tf.nn.softmax(self.logits)
            correct_preds = tf.equal(tf.argmax(preds,1),tf.argmax(self.labels,1))
            self.accuracy = tf.reduce_sum(tf.cast(correct_preds,tf.float32))


    def __build_summary(self):
        '''
        Create summaries to write on TensorBoard
        '''
        with tf.name_scope('summaries'):
            tf.summary.image('images',tf.cast(128*(self.images+1),tf.uint8),max_outputs=6)
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('accuracy', self.accuracy)
            tf.summary.histogram('histogram loss', self.loss)
            tf.summary.scalar('learning_rate',self.lr)
            self.summary_op = tf.summary.merge_all()
            val_summaries = []
            val_summaries.append(tf.summary.scalar("val_loss",self.loss))
            val_summaries.append(tf.summary.scalar('val_accuracy',self.accuracy))
            self.val_summary_op = tf.summary.merge(val_summaries)


    def build_net(self):
        self.__build_input()
        self.__build_network()
        self.__build_loss()
        self.__build_optimizer()
        self.__build_eval()
        self.__build_summary()
        print("build net finished\n")

    def eval(self,sess,epoch,writer,step):
        feed = {self.is_training:False}    
        sess.run(self.test_init)
        total_correct_preds = 0
        batches = 0
        try:
            while True:
                accuracy_batch,summaries = sess.run([self.accuracy,self.val_summary_op],feed_dict=feed)
                writer.add_summary(summaries, global_step=step)
                total_correct_preds += accuracy_batch
                batches += 1
                print('correct preds:{0},batch num:{1}'.format(total_correct_preds,batches))
                # images = np.array(images)
                # for i in range(64):
                #     plt.subplot(8,8,i+1)
                #     plt.imshow((128*(images[i]+1)).astype(np.uint8))
                #     plt.axis('off')
                # plt.show()
                #break #wrong
        except tf.errors.OutOfRangeError:
            pass
        print('Accuracy at epoch {0}: {1} '.format(epoch, total_correct_preds/10000))
        print("eval finished\n")

    def train(self):

        tfconfig = tf.ConfigProto(allow_soft_placement=True)
        tfconfig.gpu_options.allow_growth=True
        writer = tf.summary.FileWriter('./graphs/densenet', tf.get_default_graph())
        val_writer = tf.summary.FileWriter('./graphs/densenet_val',tf.get_default_graph())
        with tf.Session(config = tfconfig) as sess:

            sess.run(tf.global_variables_initializer())

            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/checkpoint'))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess,ckpt.model_checkpoint_path)
            step = self.gstep.eval()

            for epoch in range(cfg.TRAIN.epochs):
                sess.run(self.train_init,feed_dict={self.is_training:True})
                total_loss = 0
                n_batches = 0
                start_time = time.time()

                try:
                    while True:
                        _, l,summaries = sess.run([self.opt, self.loss,self.summary_op],feed_dict={self.is_training:True})
                        writer.add_summary(summaries,step)
                        if (step + 1) % cfg.TRAIN.VERBOSE == 0:
                            print('Loss at step {0}: {1}'.format(step+1, l))
                        step += 1
                        total_loss += l
                        n_batches += 1
                        #print('train:')
                        #print("logits:",np.argmax(logits,axis=1))
                        #print("labels:",np.argmax(labels,axis=1))
                        
                        #if (n_batches > 20): break   #wrong
                except tf.errors.OutOfRangeError:
                    pass
                saver.save(sess, 'checkpoints/cifar10-densenet_', step)
                print('Average loss at epoch {0}: {1}'.format(epoch, total_loss/n_batches))
                print('Took: {0} seconds'.format(time.time() - start_time))
                self.eval(sess,epoch,val_writer,step)
        writer.close()
        val_writer.close()
        print("train finished\n")

    def test(self):
        print("test finished\n")
