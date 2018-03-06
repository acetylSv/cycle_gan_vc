import tensorflow as tf
from modules import *

def build_generator(inputs):
    # inputs_shape: [batch, w=128, c=26]
    with tf.variable_scope('layer1_conv'):
        # h1_shape: [batch, w=128, c=128]
        h1 = conv1d(inputs, filters=128, size=15, dilation=1, strides=1, \
                padding="SAME", use_bias=True, activation_fn=None)
        h1 = GLU(h1)

    with tf.variable_scope('layer2_downsample'):
        # h2_shape: [batch, w=64, c=256]
        h2 = CIG_block(h1, filters=256, size=5, strides=2)

    with tf.variable_scope('layer3_downsample'):
        # h3_shape: [batch, w=32, c=512]
        h3 = CIG_block(h2, filters=512, size=5, strides=2)

    # h4-9_shape: [batch, w=32, c=512]
    with tf.variable_scope('layer4_res'):
        h4 = CIGCIS_block(h3, filters=[1024, 512], size=3, strides=1)
    with tf.variable_scope('layer5_res'):
        h5 = CIGCIS_block(h4, filters=[1024, 512], size=3, strides=1)
    with tf.variable_scope('layer6_res'):
        h6 = CIGCIS_block(h5, filters=[1024, 512], size=3, strides=1)
    with tf.variable_scope('layer7_res'):
        h7 = CIGCIS_block(h6, filters=[1024, 512], size=3, strides=1)
    with tf.variable_scope('layer8_res'):
        h8 = CIGCIS_block(h7, filters=[1024, 512], size=3, strides=1)
    with tf.variable_scope('layer9_res'):
        h9 = CIGCIS_block(h8, filters=[1024, 512], size=3, strides=1)

    with tf.variable_scope('layer10_upsample'):
        # h10_shape: [batch, w=32, c=1024] -> [batch, w=64, c=512]
        h10 = conv1d(h9, filters=1024, size=5, dilation=1, strides=1, \
                padding="SAME", use_bias=True, activation_fn=None)
        h10 = pixel_shuffler(h10)
        h10 = instance_norm(h10)
        h10 = GLU(h10)

    with tf.variable_scope('layer11_upsample'):
        # h11_shape: [batch, w=64, c=512] -> [batch, w=128, c=256]
        h11 = conv1d(h10, filters=512, size=5, dilation=1, strides=1, \
            padding="SAME", use_bias=True, activation_fn=None)
        h11 = pixel_shuffler(h11)
        h11 = instance_norm(h11)
        h11 = GLU(h11)

    with tf.variable_scope('layer12_conv'):
        # h12_shape: [batch, w=128, c=513]
        h12 = conv1d(h11, filters=hp.mcep_dim, size=15, dilation=1, strides=1, \
                padding="SAME", use_bias=True, activation_fn=None)
    return h12

def build_discriminator(inputs):
    # inputs_shape: [batch, w=128, c=513]
    # inputs_reshape_shape: [batch, h=513, w=128, c=1]
    inputs = tf.transpose(inputs, [0, 2, 1])
    inputs_reshape = tf.expand_dims(inputs, [-1])

    with tf.variable_scope('layer1_conv'):
        # h1_shape: [batch, h=513, w=64, c=128]
        h1 = conv2d(inputs_reshape, filters=128, size=[3,3], dilation=[1,1], strides=[1,2], \
                padding="SAME", use_bias=True, activation_fn=None)
        h1 = GLU(h1)

    with tf.variable_scope('layer2_downsample'):
        # h2_shape: [batch, h=257, w=32, c=256]
        h2 = CIG_block_2D(h1, filters=256, size=[3,3], strides=[2,2])
    with tf.variable_scope('layer3_downsample'):
        # h4_shape: [batch, h=129, w=16, c=512]
        h3 = CIG_block_2D(h2, filters=512, size=[3,3], strides=[2,2])
    with tf.variable_scope('layer4_downsample'):
        # h4_shape: [batch, h=129, w=8, c=1024]
        h4 = CIG_block_2D(h3, filters=1024, size=[6,3], strides=[1,2])

    with tf.variable_scope('layer5_FC'):
        # maybe need to do 1x1 conv first
        # h5_shape: [batch, 1]
        h5 = tf.layers.dense(tf.layers.flatten(h4), 1)

    with tf.variable_scope('layer6_sigmoid'):
        h6 = tf.sigmoid(h5)

    return h5, h6
