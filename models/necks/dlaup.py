import numpy as np
import tensorflow as tf 
from ..common import ConvNormActBlock
from ..builder import NECKS


def ida_up(inputs, 
           kernel_size,
           infilters, 
           outfilters, 
           up_factors,
           data_format="channels_last", 
           normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
           activation=dict(activation="relu"),
           kernel_initializer="he_normal",
           name="ida_up"):
    assert len(inputs) == len(infilters), '{} vs {} inputs'.format(len(infilters), len(inputs))

    for i, inp in enumerate(inputs):
        if infilters[i] == outfilters:
            x = inp
        else:
            x = ConvNormActBlock(outfilters,
                                 kernel_size=1,
                                 strides=1,
                                 kernel_initializer=kernel_initializer,
                                 activation=activation,
                                 normalization=normalization,
                                 name=name + "/proj_%d" % i)(inp)
        if up_factors[i] != 1:
            kernel_size = (up_factors[i] * 2, up_factors[i] * 2)
            strides = (up_factors[i], up_factors[i])
            x = tf.keras.layers.Conv2DTranspose(filters=outfilters,
                                                kernel_size=kernel_size,
                                                strides=strides,
                                                padding="same",
                                                output_padding=None,
                                                groups=outfilters,
                                                kernel_initializer=kernel_initializer,
                                                name=name + "/up_%d" % i,
                                                use_bias=False)(x)
        inputs[i] = x
    
    x = inputs[0]
    outputs = []
    channel_axis = -1 if data_format == "channels_last" else 1
    for i in range(1, len(inputs)):
        x = tf.keras.layers.Concatenate(axis=channel_axis, name=name + "/cat%d" % i)([x, inputs[i]])
        x = ConvNormActBlock(outfilters,
                             kernel_size=kernel_size,
                             strides=1,
                             kernel_initializer=kernel_initializer,
                             activation=activation,
                             normalization=normalization,
                             name=name + "/node_%d" % i)(x)
        
        outputs.append(x)
    
    return x, outputs


@NECKS.register("DLAUp")
def dla_up(filters=None, 
           input_shapes=None, 
           downsample_ratio=4,
           data_format="channels_last", 
           normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
           activation=dict(activation="relu"),
           kernel_initializer="he_normal",
           name="dla_up"):
    assert downsample_ratio in [2, 4, 8, 16]

    first_level = int(np.log2(downsample_ratio))
    scales = [2 ** i for i in range(len(filters[first_level:]))]

    inputs = [tf.keras.Input(shape=shape) for shape in input_shapes[first_level:]]

    layers = [i for i in inputs]
    infilters = filters
    scales = np.array(scales, dtype=int)
    for i in range(len(filters[first_level:]) - 1):
        j = -i - 2
        x, y = ida_up(layers[j:],
                      kernel_size=3,
                      infilters=infilters[j:],
                      outfilters=filters[j],
                      up_factors=scales[j:] // scales[j],
                      data_format=data_format,
                      normalization=normalization,
                      activation=activation,
                      kernel_initializer=kernel_initializer,
                      name=name + "/ida_%d" % i)
        scales[j+1:] = scales[j]
        infilters[j+1:] = [filters[j] for _ in filters[j+1:]]
        layers[-i-1:] = y
    
    return tf.keras.Model(inputs=inputs, outputs=x, name=name)
        
