import tensorflow as tf 
from ..common import ConvNormActBlock
from ..builder import NECKS


def bottleneck(inputs, 
               filters=512,
               midfilters=128,
               dilation_rate=1,
               normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-3, axis=-1, trainable=True),
               activation=dict(activation="swish"),
               kernel_initializer="he_normal",
               data_format="channels_last",
               name="bottleneck"):
    
    shortcut = inputs

    x = ConvNormActBlock(filters=midfilters,
                         kernel_size=1,
                         normalization=normalization,
                         activation=activation,
                         data_format=data_format,
                         use_bias=True,
                         name=name + "/conv1")(inputs)
    x = ConvNormActBlock(filters=midfilters,
                         kernel_size=3,
                         dilation_rate=dilation_rate,
                         normalization=normalization,
                         activation=activation,
                         data_format=data_format,
                         use_bias=True,
                         name=name + "/conv2")(x)
    x = ConvNormActBlock(filters=filters,
                         kernel_size=1,
                         normalization=normalization,
                         activation=activation,
                         data_format=data_format,
                         use_bias=True,
                         name=name + "/conv3")(x)
    
    x = tf.keras.layers.Add(name=name + "/add")([x, shortcut])
    
    return x


@NECKS.register("DilatedEncoder")
def DilatedEncoder(input_shapes, 
                   dilation_rates,
                   filters=512,
                   midfilters=128,
                   normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-3, axis=-1, trainable=True),
                   activation=dict(activation="swish"),
                   kernel_initializer="he_normal",
                   data_format="channels_last",
                   name="dilated_encoder"):
    inputs = tf.keras.Input(shape=input_shapes)

    x = ConvNormActBlock(filters=filters,
                         kernel_size=1,
                         data_format=data_format,
                         normalization=normalization,
                         activation=None,
                         use_bias=True,
                         name=name + "/lateral")(inputs)
    x = ConvNormActBlock(filters=filters,
                         kernel_size=3,
                         data_format=data_format,
                         normalization=normalization,
                         use_bias=True,
                         activation=None,
                         name=name + "/fpn")(x)
    
    for i, dilation_rate in enumerate(dilation_rates):
        x = bottleneck(inputs=x,
                       filters=filters,
                       midfilters=midfilters, 
                       dilation_rate=dilation_rate, 
                       normalization=normalization,
                       activation=activation,
                       kernel_initializer=kernel_initializer,
                       data_format=data_format,
                       name=name + "/dilated_encoder_blocks/%d" % i)
    
    return tf.keras.Model(inputs=inputs, outputs=x)


def make_decoder(feat_dims,
                 cls_num_convs,
                 reg_num_convs,
                 kernel_size=3,
                 data_format="channels_last",
                 use_bias=True,
                 kernel_initializer="he_normal",
                 normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-3, axis=-1, trainable=True),
                 activation=dict(activation="swish")):
    cls_subnet = tf.keras.Sequential(name="cls_subnet")
    reg_subnet = tf.keras.Sequential(name="bbox_subnet")

    for i in range(cls_num_convs):
        cls_subnet.add(ConvNormActBlock(filters=feat_dims,
                                        kernel_size=3,
                                        data_format=data_format,
                                        use_bias=True,
                                        kernel_initializer=kernel_initializer,
                                        normalization=normalization,
                                        activation=activation,
                                        name=str(i)))
    for i in range(reg_num_convs):
        reg_subnet.add(ConvNormActBlock(filters=feat_dims,
                                        kernel_size=3,
                                        data_format=data_format,
                                        use_bias=True,
                                        kernel_initializer=kernel_initializer,
                                        normalization=normalization,
                                        activation=activation,
                                        name=str(i)))

    return cls_subnet, reg_subnet