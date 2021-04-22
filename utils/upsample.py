import tensorflow as tf 


def nearest_upsample2d(inputs, factor):
    # Instead of broadcasting with a 6-d tensor, we're using stacking here
    # for TfLite compatibity.
    bs, h, w, c = tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2], tf.shape(inputs)[3]
    # bs = -1 if bs is None else bs
    data = tf.reshape(inputs, [bs, h, 1, w, 1, c]) * tf.ones([1, 1, factor, 1, factor, 1], dtype=inputs.dtype)
    
    return tf.reshape(data, [bs, h * scale, w * scale, c])
