import tensorflow as tf

def get_logreg_model(seq_batch, label_batch, seq_len, label_len):
    B = tf.get_variable("B", [seq_len, label_len], dtype=tf.float32)
    logits = tf.matmul(seq_batch, B)
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label_batch, logits=logits), axis=[0,1])
    return loss, logits
