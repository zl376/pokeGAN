# Definition of:
#   2D Pixel shuffle (for up-sampling)
#
# Credit: https://github.com/tetrachrome/subpixel
# Author: Zhe Liu (zl376@cornell.edu)
# Date: 2019-04-13

import tensorflow as tf



def pixel_shuffle(input, r):
    x = input
    bsize, a, b, c = x.get_shape().as_list()
    c = c//r//r
    bsize = tf.shape(x)[0]              # Handling Dimension(None) type for undefined batch dim
    x = tf.reshape(x, (bsize, a, b, c, r, r))
    x = tf.transpose(x, (0, 1, 2, 4, 5, 3))                      # bsize, a, b, ra, rb, c
    xs = tf.split(x, num_or_size_splits=a, axis=1)               # [bsize, 1, b, ra, rb, c] * a
    x = tf.concat([tf.squeeze(x, axis=1) for x in xs], axis=2)   # bsize, b, a*ra, rb, c
    xs = tf.split(x, num_or_size_splits=b, axis=1)               # [bsize, 1, a*ra, rb, c] * b
    x = tf.concat([tf.squeeze(x, axis=1) for x in xs], axis=2)   # bsize, a*ra, b*rb, c
    return tf.reshape(x, (bsize, a*r, b*r, c))
