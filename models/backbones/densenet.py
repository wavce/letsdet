import os
import tensorflow as tf
from models.builder import BACKBONES
from models.backbones import Backbone
from core.layers import build_convolution
from core.layers import build_normalization


class DenseNet(Backbone):
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
                 input_shape=(224, 224, 3),
                 input_tensor=None,
                 num_classes=1000,
                 drop_rate=0.5,
                 **kwargs):
        super(DenseNet, self).__init__(name=name,
                                       convolution=convolution,
                                       normalization=normalization,
                                       activation=activation,
                                       output_indices=output_indices,
                                       strides=strides,
                                       dilation_rates=dilation_rates,
                                       frozen_stages=frozen_stages,
                                       input_shape=input_shape,
                                       input_tensor=input_tensor,
                                       num_classes=num_classes,
                                       drop_rate=drop_rate,
                                       **kwargs)
        self._rgb_mean = [0.485, 0.456, 0.406]
        self._rgb_std = [0.229, 0.224, 0.225]

        self.blocks = blocks

    def dense_block(self, x, blocks, dilation_rate, drop_rate, trainable, name):
        """A dense block.

        Args:
            x: input tensor.
            blocks: integer, the number of building blocks.
            dilation_rate: integer, the dilation rate for atrous conv.
            drop_rate: float, dropout rate.
            trainable: bool, does freeze this block
            name: string, block label.

        Returns:
            output tensor for the block.
        """
        for i in range(blocks):
            x = self.conv_block(x, 32, dilation_rate, drop_rate, trainable, name + "_block" + str(i + 1))
        return x

    def transition_block(self, x, reduction, strides, trainable, name):
        """A transition block.

        Args:
            x: input tensor.
            reduction: float, compression rate at transition layers.
            strides: integer, stride in pool layer.
            trainable: bool, does freeze this block.
            name: string, block label.

        Returns:
            output tensor for the block.
        """
        bn_axis = 3 if tf.keras.backend.image_data_format() == "channels_last" else 1
        x = build_normalization(**self.normalization, name=name + "_bn")(x)
        x = tf.keras.layers.Activation(**self.activation, name=name + "_relu")(x)
        preact = x

        x = build_convolution(self.convolution,
                              filters=int(tf.keras.backend.int_shape(x)[bn_axis] * reduction),
                              kernel_size=1,
                              use_bias=False,
                              trainable=trainable,
                              name=name + "_conv")(x)
        x = tf.keras.layers.AvgPool2D(2, strides=strides, name=name + "_pool")(x)

        return preact, x

    def conv_block(self, x, growth_rate, dilation_rate, drop_rate, trainable, name):
        """A building block for a dense block.

        Args:
            x: input tensor.
            growth_rate: float, growth rate at dense layers.
            dilation_rate: integer, dilation rate.
            drop_rate: float, the dropout rate.
            trainable: bool, does freeze this block
            name: string, block label.

        Returns:
            Output tensor for the block.
        """
        bn_axis = 3 if tf.keras.backend.image_data_format() == "channels_last" else 1
        x1 = build_normalization(**self.normalization, name=name + "_0_bn")(x)
        x1 = tf.keras.layers.Activation(**self.activation, name=name + "_0_relu")(x1)

        x1 = build_convolution(self.convolution,
                               filters=4 * growth_rate,
                               kernel_size=1,
                               use_bias=False,
                               trainable=trainable,
                               name=name + "_1_conv")(x1)
        x1 = build_normalization(**self.normalization, name=name + "_1_bn")(x1)
        x1 = tf.keras.layers.Activation(**self.activation, name=name + "_1_relu")(x1)
        x1 = build_convolution(self.convolution,
                               filters=growth_rate,
                               kernel_size=3,
                               padding="same",
                               dilation_rate=dilation_rate,
                               use_bias=False,
                               trainable=trainable,
                               name=name + "_2_conv")(x1)
        x1 = tf.keras.layers.Dropout(rate=drop_rate)(x1)
        x = tf.keras.layers.Concatenate(axis=bn_axis, name=name + "_concat")([x, x1])

        return x

    def build_model(self):
        inputs = tf.keras.layers.Input((None, None, 3))
        trainable = 1 not in self.frozen_stages
        x = tf.keras.layers.Lambda(function=lambda inp: ((inp * (1. / 255.)) - self._rgb_mean) / self._rgb_std)(inputs)
        x = tf.keras.layers.ZeroPadding2D(padding=((3, 3), (3, 3)))(x)
        x = build_convolution(self.convolution,
                              filters=64,
                              kernel_size=7,
                              strides=self.strides[0],
                              use_bias=False,
                              trainable=trainable,
                              name="conv1/conv")(x)
        x = build_normalization(**self.normalization, name="conv1/bn")(x)
        x1 = tf.keras.layers.Activation(**self.activation, name="conv1/relu")(x)

        trainable = 2 not in self.frozen_stages
        x = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
        x = tf.keras.layers.MaxPool2D(3, strides=self.strides[1], name="pool1")(x)
        x = self.dense_block(x,
                             blocks=self.blocks[0],
                             dilation_rate=self.dilation_rates[1],
                             drop_rate=self.drop_rate,
                             trainable=trainable,
                             name="conv2")

        trainable = 3 not in self.frozen_stages
        preact2, x = self.transition_block(x, 0.5, self.strides[2], trainable, "pool2")
        x = self.dense_block(x,
                             blocks=self.blocks[1],
                             dilation_rate=self.dilation_rates[2],
                             drop_rate=self.drop_rate,
                             trainable=trainable,
                             name="conv3")

        trainable = 4 not in self.frozen_stages
        preact3, x = self.transition_block(x, 0.5, self.strides[3], trainable, "pool3")
        x = self.dense_block(x,
                             blocks=self.blocks[2],
                             dilation_rate=self.dilation_rates[3],
                             drop_rate=self.drop_rate,
                             trainable=trainable,
                             name="conv4")

        trainable = 5 not in self.frozen_stages
        preact4, x = self.transition_block(x, 0.5, self.strides[4], trainable, "pool4")
        x = self.dense_block(x,
                             blocks=self.blocks[3],
                             dilation_rate=self.dilation_rates[4],
                             drop_rate=self.drop_rate,
                             trainable=trainable,
                             name="conv5")
        x = build_normalization(**self.normalization, name="bn")(x)
        x5 = tf.keras.layers.Activation(**self.activation, name="relu")(x)

        if self._is_classifier:
            x = tf.keras.layers.GlobalAvgPool2D(name="avg_pool")(x5)
            x = tf.keras.layers.Dense(self.num_classes, name="fc1000")(x)

            return tf.keras.Model(inputs=inputs, outputs=x)

        else:
            outputs = [o for i, o in enumerate([x1, preact2, preact3, preact4, x5]) if i+1 in self.output_indices]

            return tf.keras.Model(inputs=inputs, outputs=outputs, name=self.name)


@BACKBONES.register("DenseNet121")
def DenseNet121(convolution="conv2d",
                normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                activation=dict(activation="relu"),
                output_indices=(-1, ),
                strides=(2, 2, 2, 2, 2),
                dilation_rates=(1, 1, 1, 1, 1),
                frozen_stages=(-1,),
                input_shape=(224, 224, 3),
                input_tensor=None,
                num_classes=1000,
                drop_rate=0.5,
                **kwargs):
    return DenseNet(name="densenet121",
                    blocks=[6, 12, 24, 16],
                    convolution=convolution,
                    normalization=normalization,
                    activation=activation,
                    output_indices=output_indices,
                    strides=strides,
                    dilation_rates=dilation_rates,
                    frozen_stages=frozen_stages,
                    input_shape=input_shape,
                    input_tensor=input_tensor,
                    num_classes=num_classes,
                    drop_rate=drop_rate,
                    **kwargs).build_model()


@BACKBONES.register
def DenseNet169(convolution="conv2d",
                normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                activation=dict(activation="relu"),
                output_indices=(-1, ),
                strides=(2, 2, 2, 2, 2),
                dilation_rates=(1, 1, 1, 1, 1),
                frozen_stages=(-1,),
                input_shape=(224, 224, 3),
                input_tensor=None,
                num_classes=1000,
                drop_rate=0.5,
                **kwargs):
    return DenseNet(name="densenet169",
                    blocks=[6, 12, 32, 32],
                    convolution=convolution,
                    normalization=normalization,
                    activation=activation,
                    output_indices=output_indices,
                    strides=strides,
                    dilation_rates=dilation_rates,
                    frozen_stages=frozen_stages,
                    input_shape=input_shape,
                    input_tensor=input_tensor,
                    num_classes=num_classes,
                    drop_rate=drop_rate,
                    **kwargs).build_model()


@BACKBONES.register
def DenseNet201(convolution="conv2d",
                normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                activation=dict(activation="relu"),
                output_indices=(-1, ),
                strides=(2, 2, 2, 2, 2),
                dilation_rates=(1, 1, 1, 1, 1),
                frozen_stages=(-1,),
                input_shape=(224, 224, 3),
                input_tensor=None,
                num_classes=1000,
                drop_rate=0.5,
                **kwargs):
    return DenseNet(name="densenet201",
                    blocks=[6, 12, 48, 32],
                    convolution=convolution,
                    normalization=normalization,
                    activation=activation,
                    output_indices=output_indices,
                    strides=strides,
                    dilation_rates=dilation_rates,
                    frozen_stages=frozen_stages,
                    input_shape=input_shape,
                    input_tensor=input_tensor,
                    num_classes=num_classes,
                    drop_rate=drop_rate,
                    **kwargs).build_model()

if __name__ == "__main__":
    densenet = DenseNet201()
    densenet.load_weights("G:/Papers/pretrained_weights/densenet201.h5")

    with tf.io.gfile.GFile("G:/Papers/panda.jpg", "rb") as gf:
        images = tf.image.decode_jpeg(gf.read())

    images = tf.image.resize(images, (224, 224))
    images = tf.expand_dims(images, axis=0)
    cls = densenet(images, training=False)

    print(tf.argmax(tf.squeeze(cls, axis=0)))
    print(tf.reduce_max(cls))

    densenet.save_weights("G:/Papers/pretrained_weights/densenet201/densenet201.ckpt")
