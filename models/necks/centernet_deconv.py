import math
import numpy as np
import tensorflow as tf 
from ..builder import NECKS
from ..common import ConvNormActBlock
from core.layers import build_activation
from core.layers import build_normalization


def _deconv_init(shape):
    w = np.random.uniform(-0.01, 0.01, size=shape)
    shape = w.shape
    f = math.ceil(shape[0] / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(shape[0]):
        for j in range(shape[1]):
            w[0, 0, i, j] = (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, shape[3]):
        w[:, :, 0, c] = w[:, :, 0, 0]
    
    return w


def deconv(inputs, 
           filters,
           deconv_kernel_size,
           deconv_strides=2,
           deconv_padding="same",
           deconv_output_padding=None,
           data_format="channels_last", 
           normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
           activation=dict(activation="relu"),
           kernel_initializer="he_normal",
           groups=1,
           dilation_rate=1,
           name="deconv"):
    x = ConvNormActBlock(filters=filters,
                         kernel_size=3,
                         dilation_rate=dilation_rate,
                         normalization=normalization,
                         activation=activation,
                         kernel_initializer=kernel_initializer,
                         use_bias=True,
                         name=name + "/conv")(inputs)
    
    deconv_kernel_shape = [deconv_kernel_size, deconv_kernel_size, filters, filters]
    x = tf.keras.layers.Conv2DTranspose(filters=filters,
                                        kernel_size=deconv_kernel_size,
                                        strides=deconv_strides,
                                        padding=deconv_padding,
                                        output_padding=deconv_output_padding,
                                        use_bias=False,
                                        kernel_initializer=tf.keras.initializers.Constant(_deconv_init(deconv_kernel_shape)),
                                        name=name + "/up_sample")(x)
    x = build_normalization(name=name + "/up_bn", **normalization)(x)
    x = build_activation(name=name + "/relu", **activation)(x)

    return x



def dcn_deconv(inputs, 
               filters,
               deconv_kernel_size,
               deconv_strides=2,
               deconv_padding="same",
               deconv_output_padding=None,
               data_format="channels_last", 
               normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
               activation=dict(activation="relu"),
               kernel_initializer="he_normal",
               groups=1,
               dilation_rate=1,
               name="deconv"):
    x = ConvNormActBlock(filters=filters,
                         kernel_size=3,
                         dilation_rate=dilation_rate,
                         normalization=normalization,
                         activation=activation,
                         kernel_initializer=kernel_initializer,
                         use_bias=True,
                         name=name + "/conv")(inputs)
    
    deconv_kernel_shape = [deconv_kernel_size, deconv_kernel_size, filters, filters]
    x = tf.keras.layers.Conv2DTranspose(filters=filters,
                                        kernel_size=deconv_kernel_size,
                                        strides=deconv_strides,
                                        padding=deconv_padding,
                                        output_padding=deconv_output_padding,
                                        use_bias=False,
                                        kernel_initializer=tf.keras.initializers.Constant(_deconv_init(deconv_kernel_shape)),
                                        name=name + "/up_sample")(x)
    x = build_normalization(name=name + "/up_bn", **normalization)(x)
    x = build_activation(name=name + "/relu", **activation)(x)

    return x



@NECKS.register("CenterNetDeconv")
def centernet_deconv(deconv_kernel_sizes=[4, 4, 4],
                     normalization=None,
                     activation=None,
                     dropblock=None,
                     data_format="channels_last",
                     kernel_initializer="he_normal",
                     input_shapes=None,
                     name="neck"):
    inputs = [tf.keras.Input(shape=shape) for shape in input_shapes]
    infilters = [shape[-1] if data_format == "channels_last" else shape[0] for shape in input_shapes]
   
    x = deconv(inputs=inputs[-1], 
               filters=infilters[-2],
               deconv_kernel_size=deconv_kernel_sizes[-1],
               normalization=normalization,
               activation=activation,
               name=name + "/deconv1")
    l = tf.keras.layers.Conv2D(filters=infilters[-2],
                               kernel_size=1,
                               kernel_initializer=kernel_initializer,
                               name=name + "/lateral_conv1")(inputs[-2])
    x = tf.keras.layers.Add(name=name + "/add1")([x, l])

    x = deconv(inputs=x, 
               filters=infilters[-3],
               deconv_kernel_size=deconv_kernel_sizes[-2],
               normalization=normalization,
               activation=activation,
               name=name + "/deconv2")
    l = tf.keras.layers.Conv2D(filters=infilters[-3],
                               kernel_size=1,
                               kernel_initializer=kernel_initializer,
                               name=name + "/lateral_conv2")(inputs[-3])
    x = tf.keras.layers.Add(name=name + "/add2")([x, l])
    
    x = deconv(inputs=x, 
               filters=infilters[-4],
               deconv_kernel_size=deconv_kernel_sizes[-3],
               normalization=normalization,
               activation=activation,
               name=name + "/deconv3")
    l = tf.keras.layers.Conv2D(filters=infilters[-4],
                               kernel_size=1,
                               kernel_initializer=kernel_initializer,
                               name=name + "/lateral_conv3")(inputs[-4])
    x = tf.keras.layers.Add(name=name + "/add3")([x, l])

    x = tf.keras.layers.Conv2D(filters=infilters[-4],
                               kernel_size=3,
                               strides=1,
                               padding="same",
                               data_format=data_format,
                               kernel_initializer=kernel_initializer,
                               name=name + "/output_conv")(x)
    
    return tf.keras.Model(inputs=inputs, outputs=x)
    
