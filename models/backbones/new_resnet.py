import tensorflow as tf
from backbones.backbone import Backbone
from core.layers import build_convolution
from core.layers import build_normalization


class BasicBlock(tf.keras.Model):
    def __init__(self,
                 convolution,
                 filters,
                 strides=1,
                 dilation_rate=1,
                 normalization="group_norm",
                 group=16,
                 activation="relu",
                 weight_decay=0.,
                 use_conv_shortcut=False,
                 **kwargs):
        super(BasicBlock, self).__init__(**kwargs)

        axis = 3 if tf.keras.backend.image_data_format() == "channels_last" else 1

        self.conv1 = build_convolution(convolution,
                                       filters=filters,
                                       kernel_size=3,
                                       strides=strides,
                                       padding="same",
                                       dilation_rate=dilation_rate if strides == 1 else 1,
                                       kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                       use_bias=False)
        self.norm1 = build_normalization(normalization, axis=axis, group=group)
        self.act = tf.keras.layers.Activation(activation)
        self.conv2 = build_convolution(convolution,
                                       filters=filters,
                                       kernel_size=3,
                                       strides=1,
                                       padding="same",
                                       dilation_rate=dilation_rate,
                                       kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                       use_bias=False)
        self.norm2 = build_normalization(normalization, axis=axis, group=group)

        if use_conv_shortcut:
            if strides >= 2:
                self.avg_pool = tf.keras.layers.AvgPool2D(2, strides, "same")
            self.conv3 = build_convolution(convolution,
                                           filters=filters,
                                           kernel_size=1,
                                           strides=1,
                                           padding="same",
                                           kernel_regularizer=tf.keras.regularizers.l2(weight_decay))
            self.norm3 = build_normalization(normalization, axis=axis, group=group)

        self.use_conv_shortcut = use_conv_shortcut

    def call(self, inputs, training=None):
        x = self.conv1(inputs)
        x = self.norm1(x, training=training)
        x = self.act(x)
        x = self.conv1(x)
        x = self.norm2(x, training=training)

        shortcut = inputs
        if self.use_conv_shortcut:
            if hasattr(self, "avg_pool"):
                shortcut = self.avg_pool(shortcut)
            shortcut = self.conv3(shortcut)
            shortcut = self.norm3(shortcut, training=training)

        x += shortcut
        x = self.act(x)

        return x


class NewResNet18(Backbone):
    def __init__(self,
                 convolution="conv2d",
                 normalization="batch_norm",
                 activation="relu",
                 output_indices=(-1, ),
                 strides=(2, 2, 2, 2, 2),
                 dilation_rates=(1, 1, 1, 1, 1),
                 freezing_stages=(-1, ),
                 freezing_batch_normalization=False,
                 group=32,
                 weight_decay=0.):
        super(NewResNet18, self).__init__(convolution=convolution,
                                          normalization=normalization,
                                          activation=activation,
                                          output_indices=output_indices,
                                          strides=strides,
                                          dilation_rates=dilation_rates,
                                          freezing_stages=freezing_stages,
                                          freezing_batch_normalization=freezing_batch_normalization,
                                          weight_decay=weight_decay)
        axis = 3 if tf.keras.backend.image_data_format() == "channels_last" else 1
        self.conv1_1 = tf.keras.Sequential([build_convolution(convolution,
                                                              filters=16,
                                                              kernsl_size=7,
                                                              padding="same",
                                                              dilation_rate=dilation_rates[0],
                                                              kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                                              use_bias=False),
                                            build_normalization(normalization, asix=axis, group=group),
                                            tf.keras.layers.Activation(activation)])
        self.conv1_2 = BasicBlock(convolution=convolution,
                                  filters=16,
                                  strides=1,
                                  dilation_rate=dilation_rates[0],
                                  normalization=normalization,
                                  group=16,
                                  activation=activation,
                                  weight_decay=weight_decay,
                                  use_conv_shortcut=False)
        self.conv1_3 = BasicBlock(convolution=convolution,
                                  filters=32,
                                  strides=strides[0],
                                  dilation_rate=dilation_rates[0],
                                  normalization=normalization,
                                  group=16,
                                  activation=activation,
                                  weight_decay=weight_decay,
                                  use_conv_shortcut=True)

        self.conv2_1 = BasicBlock(convolution=convolution,
                                  filters=64,
                                  strides=strides[1],
                                  dilation_rate=dilation_rates[1],
                                  normalization=normalization,
                                  group=16,
                                  activation=activation,
                                  weight_decay=weight_decay,
                                  use_conv_shortcut=True)
        self.conv2_2 = BasicBlock(convolution=convolution,
                                  filters=64,
                                  strides=1,
                                  dilation_rate=dilation_rates[1],
                                  normalization=normalization,
                                  group=16,
                                  activation=activation,
                                  weight_decay=weight_decay,
                                  use_conv_shortcut=False)

        self.conv3_1 = BasicBlock(convolution=convolution,
                                  filters=128,
                                  strides=strides[2],
                                  dilation_rate=dilation_rates[2],
                                  normalization=normalization,
                                  group=16,
                                  activation=activation,
                                  weight_decay=weight_decay,
                                  use_conv_shortcut=True)
        self.conv3_2 = BasicBlock(convolution=convolution,
                                  filters=128,
                                  strides=1,
                                  dilation_rate=dilation_rates[2],
                                  normalization=normalization,
                                  group=16,
                                  activation=activation,
                                  weight_decay=weight_decay,
                                  use_conv_shortcut=False)

        self.conv4_1 = BasicBlock(convolution=convolution,
                                  filters=256,
                                  strides=strides[3],
                                  dilation_rate=dilation_rates[3],
                                  normalization=normalization,
                                  group=16,
                                  activation=activation,
                                  weight_decay=weight_decay,
                                  use_conv_shortcut=True)
        self.conv4_2 = BasicBlock(convolution=convolution,
                                  filters=256,
                                  strides=1,
                                  dilation_rate=dilation_rates[3],
                                  normalization=normalization,
                                  group=16,
                                  activation=activation,
                                  weight_decay=weight_decay,
                                  use_conv_shortcut=False)

        self.conv5_1 = BasicBlock(convolution=convolution,
                                  filters=512,
                                  strides=strides[4],
                                  dilation_rate=dilation_rates[4],
                                  normalization=normalization,
                                  group=16,
                                  activation=activation,
                                  weight_decay=weight_decay,
                                  use_conv_shortcut=True)
        self.conv5_2 = BasicBlock(convolution=convolution,
                                  filters=512,
                                  strides=1,
                                  dilation_rate=dilation_rates[4],
                                  normalization=normalization,
                                  group=16,
                                  activation=activation,
                                  weight_decay=weight_decay,
                                  use_conv_shortcut=False)

    def call(self, inputs, training=None, mask=None):
        x = self.conv1_1(inputs, training=training)
        x = self.conv1_2(x, training=training)
        x1 = self.conv1_3(x, training=training)

        x = self.conv2_1(x1, training=training)
        x2 = self.conv2_2(x, training=training)

        x = self.conv3_1(x2, training=training)
        x3 = self.conv3_2(x, training=training)

        x = self.conv4_1(x3, training=training)
        x4 = self.conv4_2(x, training=training)

        x = self.conv5_1(x4, training=training)
        x5 = self.conv5_2(x, training=training)

        return (o for i, o in enumerate([x1, x2, x3, x4, x5]) if i+1 in self.output_indices)

