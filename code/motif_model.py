import tensorflow as tf
import pandas as pd
import numpy as np

def get_motifs(motifdir, filterlength=None):
    import glob
    filters = [
            pd.read_csv(f, sep='\t', skiprows=1, header=None, names=['A','C','G','T'])
            for f in glob.glob(motifdir+'/*.pwm')
            ]
    if filterlength is None:
        filterlength = max(map(len, filters))
    print('max filter length set to', filterlength)
    filters = np.array([
            pd.concat([f,
                pd.DataFrame(np.zeros((max(0,filterlength-len(f)), 4)), columns=['A','C','G','T'])]).T.values
            for f in filters
            ])
    filters = np.array([filters.transpose()])
    return filters
# 1, 21, 4, 14


def create_new_conv_layer(input_data, num_input_channels, num_filters, filter_shape, pool_shape, pool_stride, name, set_weights=None):
    # setup the filter input shape for tf.nn.conv_2d
    conv_filt_shape = [filter_shape[0], filter_shape[1], num_input_channels,
                      num_filters]

    # initialise weights and bias for the filter
    if set_weights is None:
        weights = tf.Variable(tf.truncated_normal(conv_filt_shape, stddev=0.03),
                                          name=name+'_W')
    elif set_weights == 'dumb':
        print(filter_shape[1])
        weights_ones = np.random.choice(num_input_channels, size=(filter_shape[1], num_filters))
        weights_full = np.zeros(conv_filt_shape)
        for i in range(filter_shape[1]):
            for j in range(num_filters):
                weights_full[0, i, weights_ones[i,j], j] = 1
        weights = tf.constant(weights_full.astype(np.float32), name=name+'_W')
        # weights = tf.constant(np.ones(conv_filt_shape).astype(np.float32), name=name+'_W')
        # weights = tf.constant(np.zeros(conv_filt_shape).astype(np.float32), name=name+'_W')
    else:
        weights = tf.constant(set_weights.astype(np.float32), name=name+'_W')

    bias = tf.Variable(tf.truncated_normal([num_filters]), name=name+'_b')

    # setup the convolutional layer operation
    out_layer = tf.nn.conv2d(input_data, weights, [1, 1, 1, 1], padding='SAME')

    # add the bias
    out_layer += bias

    # apply a ReLU non-linear activation
    out_layer = tf.nn.relu(out_layer)

    # now perform max pooling
    ksize = [1, pool_shape[0], pool_shape[1], 1]
    strides = [1, pool_stride, pool_stride, 1]
    out_layer = tf.nn.max_pool(out_layer, ksize=ksize, strides=strides, padding='SAME')

    return out_layer, weights

def get_motif_model(seq_len, label_len, infos, filters):
    # infos is tuples of num_filters (i.e. # of motifs), filter_shape (i.e. ~4 x 20), pool_shape (i.e. 2 x 2), num_input_channels

    with tf.name_scope('data'):
        X = tf.placeholder(tf.float32, shape=[None, seq_len], name='Input-Sequences')
        Y = tf.placeholder(tf.float32, shape=[None, label_len], name='Output-Labels')
    
    NUM = 4
    
    reshaped_seq_batch = tf.reshape(X, (-1, 1, seq_len/NUM, NUM))

#    reshaped_seq_batch = tf.reshape(X, (-1, seq_len/NUM, NUM))
#    reshaped_seq_batch = tf.expand_dims(reshaped_seq_batch, 1)

    import math
    l = seq_len / NUM
    layer = reshaped_seq_batch
    for (layer_idx, (num_filters, filter_shape, pool_shape, pool_stride, num_input_channels)) in enumerate(infos):
        if layer_idx == 0:
            layer, weights = create_new_conv_layer(layer, num_input_channels, num_filters, filter_shape, pool_shape, pool_stride, 'layer_%d' % layer_idx,
                    set_weights=filters)
        else:
            layer, weights = create_new_conv_layer(layer, num_input_channels, num_filters, filter_shape, pool_shape, pool_stride, 'layer_%d' % layer_idx)
        if layer_idx == 0:
            input_weights = weights
        l = math.ceil((l - pool_shape[0]) / float(pool_stride)) + 1
        print(l)

#    layer_shape = tf.slice(tf.shape(layer), [1], [3])
#    flattened_len = tf.reduce_prod(layer_shape)
#    flattened_layer = tf.reshape(layer, tf.stack([tf.constant(-1),flattened_len]))
    
#    flattened_layer = tf.reshape(layer, [-1,flattened_len])

    print seq_len, 'h', (seq_len / NUM) / (4**len(infos)), l
    #flattened_len = ((seq_len / NUM) / (4**len(infos))) * infos[-1][0]
    flattened_len = int(l) * infos[-1][0]
#    pdb.set_trace()
#    flattened_len = ((seq_len / NUM) / (2**len(infos))) * infos[-1][0]
    flattened_layer = tf.reshape(layer, [-1, flattened_len])
    
#    print 'asdf', flattened_len
#    print 'gg', tf.stack([flattened_len, tf.constant(label_len)])
#    B = tf.Variable(tf.random_normal(tf.stack([flattened_len, tf.constant(label_len)])), dtype=tf.float32)
#    b = tf.Variable(tf.random_normal(tf.stack([tf.constant(1), tf.constant(label_len)])), dtype=tf.float32)
    
    #B = tf.get_variable('B', tf.stack([flattened_len, tf.constant(label_len)]), dtype=tf.float32)
    #b = tf.get_variable('b', tf.stack([tf.constant(1), tf.constant(label_len)]), dtype=tf.float32)
    
    B = tf.get_variable('B', [flattened_len, label_len], dtype=tf.float32)
    b = tf.get_variable('b', [1, label_len], dtype=tf.float32)

    logits = tf.matmul(flattened_layer, B)
    logits = logits + b

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=logits), axis=[0,1])

    return X, Y, loss, logits
