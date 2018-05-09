"""
  This module contains ResNet code which does not directly build layers. This
includes dataset management, hyperparameter and optimizer code, and argument
parsing. Code for defining the ResNet layers can be found in resnet_model.py.
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow import flags
import tensorflow as tf

import resnet_model

def process_record_dataset(dataset,is_training,batch_size,shuffle_buffer,
                           parse_record_fn,num_epochs=1):
    dataset = dataset.prefetch(buffer_size = batch_size) #使得CPU提供数据和GPU训练并行完成
    if is_training:
        dataset = dataset.shuffle(buffer_size = shuffle_buffer)

    dataset = dataset.repeat(num_epochs)

    dataset = dataset.apply(  #==dataset.batch.map
        tf.contrib.data.map_and_batch(lambda value:parse_record_fn(value,is_training),
                                      batch_size=batch_size,
                                      num_parallel_batches = 1))
    
    dataset.prefetch(buffer_size = -1)
    return dataset

def learning_rate_with_decay(batch_size,batch_denom,num_images,boundary_epochs,decay_rates):

    initial_learning_rate = 0.1 * batch_size / batch_denom
    batches_per_epoch = num_images/batch_size

    boundaries = [int(batches_per_epoch *epoch) for epoch in boundary_epochs]

    vals = [initial_learning_rate * decay for decay in decay_rates]
    def learning_rate_fn(global_step):
        global_step = tf.cast(global_step,tf.int32)
        return tf.train.piecewise_constant(global_step,boundaries,vals)
        #global_step = tf.Variable(0, trainable=False)
        #boundaries = [100000, 110000]
        #values = [1.0, 0.5, 0.1]   
        #learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
        #Example: use a learning rate that's 1.0 for the first 100001 steps, 0.5 for the next 10000 steps, and 0.1 for any additional steps.
    return learning_rate_fn

def resnet_model_fn(features,labels,mode,model_class,resnet_size,
                    weight_decay,learning_rate_fn,momentum,
                    data_format,resnet_version,loss_scale,
                    loss_filter_fn=None,dtype=resnet_model.DEFAULT_DTYPE):
    
    tf.summary.image('images',features,max_outputs=6)

    features = tf.cast(features,dtype)

    model = model_class(resnet_size,data_format,resnet_version=resnet_version,
                        dtype = dtype)
    logits = model(features, mode == tf.estimator.ModeKeys.TRAIN)

    logits = tf.cast(logits,tf.float32)

    predictions = {
        'classes':tf.argmax(logits,axis=1),
        'probabilities': tf.nn.softmax(logits,name='softmax_tensor')
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode = mode,predictions = predictions,
                                          export_outputs={
                                              'predict':tf.estimator.export.PredictOutput(predictions)
                                          })
    
    cross_entropy = tf.losses.softmax_cross_entropy(
        logits = logits,onehot_labels=labels)
    tf.identity(cross_entropy,name='cross_entropy')
    tf.summary.scalar('cross_entropy',cross_entropy)


    def exclude_batch_norm(name):    ##??
        return 'batch_normalization' not in name
    loss_filter_fn = loss_filter_fn or exclude_batch_norm


    l2_loss = weight_decay * tf.add_n(
        [tf.nn.l2_loss(tf.cast(v,tf.float32)) for v in tf.trainable_variables()
         if loss_filter_fn(v.name)])
    
    tf.summary.scalar('l2_loss',l2_loss)
    loss = cross_entropy + l2_loss

    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()

        learning_rate = learning_rate_fn(global_step)

        tf.identity(learning_rate,name='learning_rate')
        tf.summary.scalar('learning_rate',learning_rate)

        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                                momentum = momentum)
        
        if loss_scale != 1:
            scaled_grad_vars = optimizer.compute_gradients(loss * loss_scale)

            unscaled_grad_vars = [(grad/loss_scale,var) 
                                    for grad,var in scaled_grad_vars]

            minimize_op = optimizer.apply_gradients(unscaled_grad_vars,global_step)
        else:
            minimize_op = optimizer.minimize(loss,global_step)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = tf.group(minimize_op,update_ops)

    else:
        train_op = None

    accuracy = tf.metrics.accuracy(tf.argmax(labels,axis = 1),predictions['classes'])

    metrics = {'accuracy':accuracy}

    tf.identity(accuracy[1],name = 'train_accuracy')

    tf.summary.scalar('train_accuracy',accuracy[1])

    return tf.estimator.EstimatorSpec(
        mode = mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=metrics
    )

def resnet_main(
    flags_obj,model_function,input_function,dataset_name,shape = None):

    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
    session_config = tf.ConfigProto(
        allow_soft_placement=True)
    

    run_config = tf.estimator.RunConfig(session_config=session_config)
    classifier = tf.estimator.Estimator(
        model_fn = model_function, model_dir = flags_obj.model_dir,config=run_config,
        params = {
          'resnet_size': int(flags_obj.resnet_size),
          'data_format': flags_obj.data_format,
          'batch_size': flags_obj.batch_size,
          'resnet_version': 1,
          'loss_scale': 1,
          'dtype':tf.float32
        }
    )




    def input_fn_train():
        return input_function(
            is_training=True, data_dir=flags_obj.data_dir,
            batch_size=flags_obj.batch_size,
            num_epochs=flags_obj.epochs_between_evals)

    def input_fn_eval():
        return input_function(
            is_training=False, data_dir=flags_obj.data_dir,
            batch_size= flags_obj.batch_size,
            num_epochs=1)

    total_training_cycle = (flags_obj.train_epochs //
                        flags_obj.epochs_between_evals)
    for cycle_index in range(total_training_cycle):
        tf.logging.info('Starting a training cycle: %d/%d',
                    cycle_index, total_training_cycle)

        classifier.train(input_fn=input_fn_train, 
                    max_steps=None)
        tf.logging.info('Starting to evaluate.')
        eval_results = classifier.evaluate(input_fn=input_fn_eval,
                    steps=None)
        for key in sorted(eval_results):
            if key != tf.GraphKeys.GLOBAL_STEP:
                tf.summary.scalar(key,eval_results[key])


