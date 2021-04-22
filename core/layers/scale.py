import tensorflow as tf 


class Scale(tf.keras.layers.Layer):
    def __init__(self, value, **kwargs):
        super(Scale, self).__init__(**kwargs)

        self.value = value

    def build(self, input_shape):
        self.scale = self.add_weight(name="scale",
                                     trainable=True,
                                     shape=[],
                                     dtype=self.dtype,
                                     initializer=tf.keras.initializers.Constant(self.value))

    def call(self, inputs, **kwargs):
        return inputs * self.scale

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {"value": self.value}

        base_config = super(Scale, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))
