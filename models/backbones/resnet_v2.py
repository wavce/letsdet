import os
import tensorflow as tf
from .backbone import Backbone
from ..builder import BACKBONES 
from core.layers import build_convolution
from core.layers import build_normalization


def bottleneck_v2(x,
                  convolution,
                  filters,
                  strides=1,
                  dilation_rate=1,
                  normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                  activation=dict(activation="relu"),
                  trainable=True,
                  dropblock=None,
                  use_conv_shortcut=True,
                  name=None):
    """A residual block.

        Args:
            x: input tensor.
            filters: integer, filters of the bottleneck layer.
            convolution: The convolution type.
            strides: default 1, stride of the first layer.
            dilation_rate: default 1, dilation rate in 3x3 convolution.
            activation: the activation layer name.
            trainable: does this block is trainable.
            normalization: the normalization, e.g. "batch_norm", "group_norm" etc.
            dropblock: the arguments in DropBlock2D.
            use_conv_shortcut: default True, use convolution shortcut if True,
                otherwise identity shortcut.
            name: string, block label.
    Returns:
        Output tensor for the residual block.
    """
    bn_axis = 3 if tf.keras.backend.image_data_format() == "channels_last" else 1

    preact = build_normalization(**normalization, name=name + "_preact_bn")(x)
    preact = tf.keras.layers.Activation(**activation, name=name + "_preact_relu")(preact)

    if use_conv_shortcut is True:
        shortcut = build_convolution(convolution,
                                     filters=4 * filters,
                                     kernel_size=1,
                                     strides=strides,
                                     trainable=trainable,
                                     name=name + "_0_conv")(preact)
    else:
        shortcut = tf.keras.layers.MaxPooling2D(1, strides=strides)(x) if strides > 1 else x

    if dropblock is not None:
        shortcut = DropBlock2D(**dropblock, name=name + "_0_dropblock")(shortcut)

    x = build_convolution(convolution,
                          filters=filters,
                          kernel_size=1,
                          strides=1,
                          use_bias=False,
                          trainable=trainable,
                          name=name + "_1_conv")(preact)

    x = build_normalization(**normalization, name=name + "_1_bn")(x)
    x = tf.keras.layers.Activation(**activation, name=name + "_1_relu")(x)
    if dropblock is not None:
        x = DropBlock2D(**dropblock, name=name + "_1_dropblock")(x)

    x = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name=name + "_2_pad")(x)
    x = build_convolution(convolution,
                          filters=filters,
                          kernel_size=3,
                          strides=strides,
                          dilation_rate=1 if strides > 1 else dilation_rate,
                          use_bias=False,
                          trainable=trainable,
                          name=name + "_2_conv")(x)
    x = build_normalization(**normalization, name=name + "_2_bn")(x)
    x = tf.keras.layers.Activation(**activation, name=name + "_2_relu")(x)
    if dropblock is not None:
        x = DropBlock2D(**dropblock)(x)

    x = build_convolution(convolution,
                          filters=4 * filters,
                          kernel_size=1,
                          trainable=trainable,
                          name=name + "_3_conv")(x)
    if dropblock is not None:
        x = DropBlock2D(**dropblock, name=name + "_2_dropblock")(x)
    x = tf.keras.layers.Add(name=name + "_out")([shortcut, x])

    return x, preact


class ResNetV2(Backbone):
    def __init__(self,
                 name,
                 blocks,
                 convolution="conv2d",
                 normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                 activation=dict(activation="relu"),
                 output_indices=(-1, ),
                 strides=(2, 2, 2, 2, 2),
                 dilation_rates=(1, 1, 1, 1, 1),
                 frozen_stages=(-1,),
                 weight_decay=0.,
                 dropblock=None,
                 num_classes=1000,
                 drop_rate=0.5,
                 input_shape=(224, 224, 3),
                 input_tensor=None,
                 **kwargs):
        super(ResNetV2, self).__init__(name=name,
                                       convolution=convolution,
                                       normalization=normalization,
                                       activation=activation,
                                       output_indices=output_indices,
                                       strides=strides,
                                       dilation_rates=dilation_rates,
                                       frozen_stages=frozen_stages,
                                       dropblock=dropblock,
                                       num_classes=num_classes,
                                       input_shape=input_shape,
                                       input_tensor=input_tensor,
                                       drop_rate=drop_rate,
                                       **kwargs)

        # self._rgb_mean = tf.constant([123.68, 116.78, 103.94], dtype=self.dtype)
        # self._rgb_mean = tf.constant([0.485, 0.456, 0.406], dtype=self.dtype)
        # self._rgb_std = tf.constant([0.229, 0.224, 0.225], dtype=self.dtype)

        self.blocks = blocks

    def stack(self, x, filters, strides, dilation_rate, trainable, blocks, name=None):
        x, preact = bottleneck_v2(x,
                                  convolution=self.convolution,
                                  filters=filters,
                                  strides=1,
                                  dilation_rate=dilation_rate,
                                  normalization=self.normalization,
                                  activation=self.activation,
                                  trainable=trainable,
                                  dropblock=self.dropblock,
                                  use_conv_shortcut=True,
                                  name=name + '_block1')
        for i in range(2, blocks):
            x, _ = bottleneck_v2(x,
                                 convolution=self.convolution,
                                 filters=filters,
                                 strides=1,
                                 dilation_rate=dilation_rate,
                                 normalization=self.normalization,
                                 trainable=trainable,
                                 dropblock=self.dropblock,
                                 use_conv_shortcut=False,
                                 name=name + '_block' + str(i))
        x, _ = bottleneck_v2(x,
                             convolution=self.convolution,
                             filters=filters,
                             strides=strides,
                             dilation_rate=1 if strides > 1 else dilation_rate,
                             normalization=self.normalization,
                             trainable=trainable,
                             dropblock=self.dropblock,
                             use_conv_shortcut=False,
                             name=name + '_block' + str(blocks))

        return x, preact

    def build_model(self):
        trainable = 0 not in self.frozen_stages

        def _norm(inp):
            inp = inp * 255. / 127.5 - 1.

            return inp
        
        x = tf.keras.layers.Lambda(function=_norm, name="norm_input")(self.img_input) 
        
        x = tf.keras.layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name="conv1_pad")(x)
        x1 = build_convolution(self.convolution,
                               filters=64,
                               kernel_size=7,
                               strides=self.strides[0],
                               padding="valid",
                               dilation_rate=self.dilation_rates[0],
                               trainable=trainable,
                               use_bias=True,
                               name="conv1_conv")(x)

        x = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name="pool1_pad")(x1)
        x = tf.keras.layers.MaxPool2D((3, 3), strides=self.strides[1], padding="valid", name="pool1_pool")(x)

        trainable = 1 not in self.frozen_stages
        x, _ = self.stack(x, 64, self.strides[2], self.dilation_rates[1], trainable, self.blocks[0], "conv2")

        trainable = 2 not in self.frozen_stages
        x, preact2 = self.stack(x, 128, self.strides[3], self.dilation_rates[2], trainable, self.blocks[1], "conv3")

        trainable = 3 not in self.frozen_stages
        x, preact3 = self.stack(x, 256, self.strides[4], self.dilation_rates[3], trainable, self.blocks[2], "conv4")

        trainable = 4 not in self.frozen_stages
        x, preact4 = self.stack(x, 512, 1, self.dilation_rates[4], trainable, self.blocks[3], "conv5")
        x = build_normalization(**self.normalization, name="post_bn")(x)
        x5 = tf.keras.layers.Activation(**self.activation, name="post_relu")(x)

        if -1 in self.output_indices:
            x = tf.keras.layers.GlobalAvgPool2D(name="avg_pool")(x5)
            x = tf.keras.layers.Dropout(rate=self.drop_rate)(x)
            outputs = tf.keras.layers.Dense(self.num_classes, name="probs")(x)
        else:
            outputs = [o for i, o in enumerate([x1, preact2, preact3, preact4, x5]) if i in self.output_indices]

        return tf.keras.Model(inputs=self.img_input, outputs=outputs, name=self.name)
    

@BACKBONES.register("ResNet50V2")
def ResNet50V2(convolution="conv2d",
               normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
               activation=dict(activation="relu"),
               output_indices=(-1, ),
               strides=(2, 2, 2, 2, 2),
               dilation_rates=(1, 1, 1, 1, 1),
               frozen_stages=(-1, ),
               dropblock=None,
               num_classes=1000,
               drop_rate=0.5,
               input_shape=(224, 224, 3),
               input_tensor=None,
               **kwargs):
    return ResNetV2(name="resnet50v2",
                    blocks=[3, 4, 6, 3],
                    convolution=convolution,
                    normalization=normalization,
                    activation=activation,
                    output_indices=output_indices,
                    strides=strides,
                    dilation_rates=dilation_rates,
                    frozen_stages=frozen_stages,
                    dropblock=dropblock,
                    num_classes=num_classes,
                    input_shape=input_shape,
                    input_tensor=input_tensor,
                    drop_rate=drop_rate,
                    **kwargs).build_model()


@BACKBONES.register("ResNet101V2")
def ResNet101V2(convolution="conv2d",
                normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                activation=dict(activation="relu"),
                output_indices=(-1, ),
                strides=(2, 2, 2, 2, 2),
                dilation_rates=(1, 1, 1, 1, 1),
                frozen_stages=(-1, ),
                dropblock=None,
                num_classes=1000,
                drop_rate=0.5,
                input_shape=(224, 224, 3),
                input_tensor=None,
                **kwargs):
    return ResNetV2(name="resnet101v2",
                    blocks=[3, 4, 23, 3],
                    convolution=convolution,
                    normalization=normalization,
                    activation=activation,
                    output_indices=output_indices,
                    strides=strides,
                    dilation_rates=dilation_rates,
                    frozen_stages=frozen_stages,
                    dropblock=dropblock,
                    num_classes=num_classes,
                    input_shape=input_shape,
                    input_tensor=input_tensor,
                    drop_rate=drop_rate,
                    **kwargs).build_model()


@BACKBONES.register("ResNet152V2")
def ResNet152V2(convolution="conv2d",
                normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                activation=dict(activation="relu"),
                output_indices=(-1, ),
                strides=(2, 2, 2, 2, 2),
                dilation_rates=(1, 1, 1, 1, 1),
                frozen_stages=(-1, ),
                dropblock=None,
                num_classes=1000,
                drop_rate=0.5,
                input_shape=(224, 224, 3),
                input_tensor=None,
                **kwargs):
    return ResNetV2(name="resnet152v2",
                    blocks=[3, 8, 36, 3],
                    convolution=convolution,
                    normalization=normalization,
                    activation=activation,
                    output_indices=output_indices,
                    strides=strides,
                    dilation_rates=dilation_rates,
                    frozen_stages=frozen_stages,
                    dropblock=dropblock,
                    num_classes=num_classes,
                    input_shape=input_shape,
                    input_tensor=input_tensor,
                    drop_rate=drop_rate,
                    **kwargs).build_model()



if __name__ == '__main__':
    name = "resnet50v2"
    resnet = ResNet50V2()
    resnet.load_weights("/home/bail/Workspace/pretrained_weights/%s_weights_tf_dim_ordering_tf_kernels.h5" % name)

    with tf.io.gfile.GFile("/home/bail/Documents/pandas.jpg", "rb") as gf:
        images = tf.image.decode_jpeg(gf.read())

    images = tf.image.resize(images, (224, 224))
    images = tf.expand_dims(images, axis=0)
    logits = resnet(images, training=False)
    probs = tf.nn.softmax(logits)
    print(tf.nn.top_k(probs, k=5))

    # resnet.save_weights("G:/Papers/pretrained_weights/resnet152v2/resnet152v2.ckpt")