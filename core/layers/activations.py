import tensorflow as tf 


@tf.custom_gradient
def _mish(x):
    x1 = tf.nn.tanh(tf.nn.softplus(x))
    
    def _grad(dy):
        dx = x1 + x * tf.nn.sigmoid(x) * (1 - x1 * x1)

        return dx * dy
    
    return x * x1, _grad


class Mish(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Mish, self).__init__(**kwargs)

    def call(self, inputs):
        # x = inputs * (tf.nn.tanh(tf.nn.softplus(inputs)))
        
        return _mish(inputs)


