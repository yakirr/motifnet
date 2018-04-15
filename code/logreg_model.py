import tensorflow as tf

def get_logreg_model(seq_len, label_len):
    with tf.name_scope('data'):
        X = tf.placeholder(tf.float32, shape=[None, seq_len], name='Input-Sequences')
        Y = tf.placeholder(tf.float32, shape=[None, label_len], name='Output-Labels')
        B = tf.get_variable('B', [seq_len, label_len], dtype=tf.float32)

    with tf.name_scope('model'):
        logits = tf.matmul(X, B, name='logits')

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=logits),
                axis=[0,1],
                name='Loss')

    return X, Y, loss, logits
