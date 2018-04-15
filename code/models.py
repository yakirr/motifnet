import tensorflow as tf

def get_logreg_model(seq_batch, label_batch, seq_len, label_len):
    B = tf.get_variable("B", [seq_len, label_len], dtype=tf.float32)
    logits = tf.matmul(seq_batch, B)
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label_batch, logits=logits), axis=[0,1])
    return loss, logits

def create_new_conv_layer(input_data, num_input_channels, num_filters, filter_shape, pool_shape, name):
    # setup the filter input shape for tf.nn.conv_2d
    conv_filt_shape = [filter_shape[0], filter_shape[1], num_input_channels,
                      num_filters]

    # initialise weights and bias for the filter
    weights = tf.Variable(tf.truncated_normal(conv_filt_shape, stddev=0.03),
                                      name=name+'_W')
    bias = tf.Variable(tf.truncated_normal([num_filters]), name=name+'_b')

    # setup the convolutional layer operation
    out_layer = tf.nn.conv2d(input_data, weights, [1, 1, 1, 1], padding='SAME')

    # add the bias
    out_layer += bias

    # apply a ReLU non-linear activation
    out_layer = tf.nn.relu(out_layer)

    # now perform max pooling
    ksize = [1, pool_shape[0], pool_shape[1], 1]
    strides = [1, 2, 2, 1]
    out_layer = tf.nn.max_pool(out_layer, ksize=ksize, strides=strides, padding='SAME')

    return out_layer, weights

def get_novel_model(seq_batch, label_batch, seq_len, label_len, infos):
    # infos is tuples of num_filters (i.e. # of motifs), filter_shape (i.e. ~4 x 20), pool_shape (i.e. 2 x 2)
    
    NUM = 4
    num_input_channels = 1
    
    reshaped_seq_batch = tf.reshape(seq_batch, (-1, NUM, seq_len/NUM, num_input_channels))

    layer = reshaped_seq_batch
    for (layer_idx, (num_filters, filter_shape, pool_shape)) in enumerate(infos):
        layer, weights = create_new_conv_layer(layer, num_input_channels, num_filters, filter_shape, pool_shape, 'layer_%d' % layer_idx)
        if layer_idx == 0:
            input_weights = weights

    layer_shape = tf.slice(tf.shape(layer), [1], [3])
    flattened_len = tf.reduce_prod(layer_shape)
    flattened_layer = tf.reshape(layer, [-1,flattened_len])

    B = tf.get_variable('B', [flattened_len, label_len], dtype=tf.float32)
    b = tf.get_variable('b', [1, label_len], dtype=tf.float32)

    logits = tf.matmul(seq_batch, B)
    logits = logits + b

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label_batch, logits=logits), axis=[0,1])

    return loss, logits
