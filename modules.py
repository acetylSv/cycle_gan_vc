import tensorflow as tf

from hyperparams import Hyperparams as hp

def instance_norm(inputs):
    outputs = tf.contrib.layers.instance_norm(inputs, data_format='NHWC')
    return outputs
    '''
    axis = [1,2] # for format: NHWC
    epsilon = 1e-5
    mean, var = tf.nn.moments(inputs, axis, keep_dims=True)
    outputs = (inputs - mean) / tf.sqrt(var+epsilon)

    return outputs
    '''
def conv1d(inputs, filters=None, size=1, strides=1, dilation=1,
           padding="SAME", use_bias=False, activation_fn=None):
    if padding.lower()=="causal":
        # pre-padding for causality
        pad_len = (size - 1) * dilation  # padding size
        inputs = tf.pad(inputs, [[0, 0], [pad_len, 0], [0, 0]])
        padding = "valid"

    if filters is None:
        filters = inputs.get_shape().as_list[-1]

    params = {"inputs":inputs, "filters":filters, "kernel_size":size,
              "strides": strides, "dilation_rate":dilation, "padding":padding,
              "activation":activation_fn, "use_bias":use_bias}

    outputs = tf.layers.conv1d(**params)

    return outputs

def conv2d(inputs, filters=None, size=[1,1], dilation=[1,1], strides=[1,1],
           padding="SAME", use_bias=False, activation_fn=None):
    if padding.lower()=="causal":
        # pre-padding for causality
        pad_len = (size - 1) * dilation  # padding size
        inputs = tf.pad(inputs, [[0, 0], [pad_len, 0], [pad_len, 0]])
        padding = "valid"

    if filters is None:
        filters = inputs.get_shape().as_list[-1]

    params = {"inputs":inputs, "filters":filters,
              "kernel_size":size, "strides":strides,
              "dilation_rate":dilation, "padding":padding,
              "activation":None, "use_bias":use_bias}

    outputs = tf.layers.conv2d(**params)

    return outputs

def GLU(inputs):
    with tf.variable_scope('GLU'):
        if len(inputs.get_shape().as_list()) != 4:
            inputs = tf.expand_dims(inputs, 1)
        c_size = inputs.get_shape()[-1]
        conv_w = conv2d(inputs[:,:,:,:c_size//2], filters=c_size, size=[3,3], use_bias=True)
        conv_v = conv2d(inputs[:,:,:,c_size//2:], filters=c_size, size=[3,3], use_bias=True)
        outputs = conv_w * tf.sigmoid(conv_v)

        if outputs.get_shape().as_list()[1] == 1:
            outputs = tf.squeeze(outputs, axis=1)

    return outputs

def highwaynet(inputs, num_units=None):
    '''Highway networks, see https://arxiv.org/abs/1505.00387
     Args:
        inputs: A 3D tensor of shape [N, T, W].
        num_units: An int or `None`. Specifies the number of units in the
        highway layer or uses the input size if `None`.
    Returns:
        A 3D tensor of shape [N, T, W].'''
    if not num_units:
        num_units = inputs.get_shape()[-1]
    H = tf.layers.dense(inputs, units=num_units, activation=tf.nn.relu, name="dense1")
    T = tf.layers.dense(inputs, units=num_units, activation=tf.nn.sigmoid,
        bias_initializer=tf.constant_initializer(-1.0), name="dense2")
    outputs = H*T + inputs*(1.-T)

    return outputs

def CIG_block(inputs, filters, size, strides):
    h = conv1d(inputs, filters=filters, size=size, strides=strides, dilation=1, \
            padding="SAME", use_bias=True, activation_fn=None)
    h = instance_norm(h)
    h = GLU(h)
    return h

def CIG_block_2D(inputs, filters, size, strides):
    h = conv2d(inputs, filters=filters, size=size, strides=strides, dilation=[1,1], \
        padding="SAME", use_bias=True, activation_fn=None)
    h = instance_norm(h)
    h = GLU(h)
    return h

def CIGCIS_block(inputs, filters, size, strides):
    h = conv1d(inputs, filters=filters[0], size=size, strides=strides, dilation=1, \
            padding="SAME", use_bias=True, activation_fn=None)
    h = instance_norm(h)
    h = GLU(h)
    h = conv1d(h, filters=filters[1], size=size, strides=strides, dilation=1, \
            padding="SAME", use_bias=True, activation_fn=None)
    h = instance_norm(h)
    h = h + inputs
    return h

def pixel_shuffler(inputs):
    X = tf.transpose(inputs, [2, 1, 0])  # (c, w, b)
    X = tf.batch_to_space_nd(X, [2], [[0, 0]])  # (1, c*w, b)
    X = tf.transpose(X, [2, 1, 0])
    return X

# GAN criterion
def abs_criterion(in_, target):
    return tf.reduce_mean(tf.abs(in_ - target))

def mae_criterion(in_, target):
    return tf.reduce_mean((in_-target)**2)

def sce_criterion(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
