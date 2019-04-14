# Definition of customized structures
# Author: Zhe Liu (zl376@cornell.edu)
# Date: 2019-04-13

import numpy as np
import tensorflow as tf
from .pixel_shuffle import pixel_shuffle



def generator(input, img_size, len_code, n_channel, train, reuse=False, name='actfinal', upsample='upsample'):
    # c4, c8, c16, c32, c64 = 512, 256, 128, 64, 32 # channel num
    # s4 = 4
    # output_dim = n_channel
    level = 4
    filter_base = 64
    # determine parameter for each layer
    assert all( x%(2**level) == 0 for x in img_size ), 'Image size {} need to be (enough) power of 2'.format(img_size)
    base_size = tuple( x//(2**level) for x in img_size )
    filter_sizes = [ min(512, filter_base * 2**i) for i in range(level) ]
    
    def up2(x, filter_size, name_template, upsample='upsample'):
        if upsample is 'deconv':
            # Use Deconv
            x = tf.layers.conv2d_transpose(x, filter_size, kernel_size=(5,5), strides=(2,2), padding="same",
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           name=name_template.format('deconv'))
        elif upsample is 'upsample':
            # Use Upsampling + Conv
            dims = x.get_shape()[1:1+2]
            x = tf.image.resize_images(images=x, size=dims*np.array((2,2)), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            x = tf.layers.conv2d(x, filter_size, kernel_size=(5,5), padding="same",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 name=name_template.format('conv'))
        else:
            # Use Pixel Shuffle
            x = tf.layers.conv2d(x, filter_size*2*2, kernel_size=(5,5), padding="same",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 name=name_template.format('conv'))
            x = pixel_shuffle(x, 2) 
        return x

    x = input
    with tf.variable_scope('gen') as scope:
        if reuse:
            scope.reuse_variables()
            
        x = tf.layers.dense(x, np.prod(base_size)*filter_sizes[level-1],
                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                            name='fc0')
        x = tf.reshape(x, shape=(-1,)+base_size+(filter_sizes[level-1],), name='reshape0')
        x = tf.contrib.layers.batch_norm(x, is_training=train, epsilon=1e-5, decay=0.9, updates_collections=None, scope='bn0')
        x = tf.nn.relu(x, name='act0')
        
        for i in range(level-2, -1, -1):
            filter_size = filter_sizes[i]
            name_template = '{{}}{0}'.format(level-1-i)
            
            x = up2(x, filter_size, name_template, upsample=upsample)
            
            x = tf.contrib.layers.batch_norm(x, is_training=train, epsilon=1e-5, decay=0.9, updates_collections=None, scope=name_template.format('bn'))
            x = tf.nn.relu(x, name=name_template.format('act'))
        
        x = up2(x, n_channel, 'preactfinal', upsample=upsample)
        x = tf.nn.tanh(x, name=name)
        return x
    
    
def discriminator(input, img_size, train, reuse=False, name='fcfinal'):
    # c2, c4, c8, c16 = 64, 128, 256, 512  # channel num: 64, 128, 256, 512
    level = 4
    filter_base = 64
    # determine parameter for each layer
    assert all( x%(2**level) == 0 for x in img_size ), 'Image size {} need to be (enough) power of 2'.format(img_size)
    filter_sizes = [ min(512, filter_base * 2**i) for i in range(level) ]
    
    x = input
    with tf.variable_scope('dis') as scope:
        if reuse:
            scope.reuse_variables()
            
        for i in range(level):
            filter_size = filter_sizes[i]
            name_template = '{{}}{0}'.format(i+1)
            
            x = tf.layers.conv2d(x, filter_size, kernel_size=(5,5), strides=(2,2), padding="same",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 name=name_template.format('conv'))
            if i > 0:
                x = tf.contrib.layers.batch_norm(x, is_training=train, epsilon=1e-5, decay=0.9, updates_collections=None, scope=name_template.format('bn'))
            x = lrelu(x, name=name_template.format('act'))
            
        x = tf.layers.flatten(x)
        x = tf.layers.dense(x, 1,
                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                            name=name)
        return x
    
def lrelu(x, leak=0.2, name=None): 
    return tf.maximum(x, leak * x, name=name)