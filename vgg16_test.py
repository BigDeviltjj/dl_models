import vgg16
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import utils

image1 = utils.load_image('./tiger.jpeg')
image2 = utils.load_image('./puzzle.jpeg')
image = np.concatenate((image1[None,:,:,:],image2[None,:,:,:]),axis = 0)

plt.figure()
with tf.device('/cpu:0'):
    with tf.Session() as sess:
#with  tf.Session(config=tf.ConfigProto(gpu_options=(tf.GPUOptions(allow_growth=True))))  as sess:
        vgg = vgg16.Vgg16("./../vgg16.npy")

        vgg.build_graph()

        sess.run(tf.global_variables_initializer())
        logits = sess.run(vgg.logits,{vgg.data_X:image})
        logits = np.array(logits)
        print(logits.shape)
        utils.print_prob(logits[0], './synset.txt')
        utils.print_prob(logits[1], './synset.txt')
