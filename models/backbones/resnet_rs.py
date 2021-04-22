import os
import sys
import numpy as np
import tensorflow as tf
from .extractor import Extractor
from .builder import EXTRACTORS
from .common import ConvNormActBlock
from .common import squeeze_excitation
from core.layers import build_activation
from core.layers import build_normalization
from core.layers import DropConnect, get_drop_connect_rate

sys.setrecursionlimit(1000000)


def basic_block(inputs,
                filters,
                strides=1,
                dilation_rate=1,
                data_format="channels_last",
                kernel_initializer=tf.keras.initializers.VarianceScaling(),
                normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-3, axis=-1, trainable=True),
                init_gamma_zero=True,
                activation=dict(activation="relu"),
                trainable=True,
                dropblock=None,
                use_conv_shortcut=False,
                expansion=1,
                drop_connect_rate=None,
                se_ratio=None,
                preactivation=False,
                use_resnetd_shortcut=True,
                name=None,
                **kwargs):
    """
    Basic Residual block
    
    Args:
        filters(int): integer, filters of the bottleneck layer.
        strides(int): default 1, stride of the first layer.
        dilation_rate(int): default 1, dilation rate in 3x3 convolution.
        data_format(str): default channels_last,
        normalization(dict): the normalization name and hyper-parameters, e.g.
            dict(normalization="batch_norm", momentum=0.9, epsilon=1e-3, axis=-1, trainable=True), 
            dict(normalization="group_norm", epsilon=1e-3, axis=-1) etc.
        activation: the activation layer name.
        trainable: does this block is trainable.
        dropblock: the arguments in DropBlock2D
        use_conv_shortcut: default True, use convolution shortcut if True,
            otherwise identity shortcut.
        name: string, block label, default None.
    """
    if preactivation:
        x = build_normalization(name=name + "/norm1", **normalization)(inputs)
        x = build_activation(name=name + "/%s" % activation["activation"], **activation)(x)
    else:
        x = inputs

    shortcut = x
    x = ConvNormActBlock(filters=filters,
                         kernel_size=3,
                         strides=strides,
                         data_format=data_format,
                         dilation_rate=1 if strides > 1 else dilation_rate,
                         kernel_initializer=kernel_initializer,
                         trainable=trainable,
                         normalization=normalization,
                         activation=activation,
                         dropblock=dropblock,
                         name=name + "/conv1")(x)
    x = ConvNormActBlock(filters=filters,
                         kernel_size=3,
                         strides=1,
                         data_format=data_format,
                         dilation_rate=dilation_rate,
                         kernel_initializer=kernel_initializer,
                         trainable=trainable,
                         normalization=None if preactivation else normalization,
                         activation=None,
                         dropblock=dropblock,
                         name=name + "/conv2")(x)
    
    if use_conv_shortcut:
        if use_resnetd_shortcut and strides > 1:
            shortcut = tf.keras.layers.AvgPool2D(pool_size=strides, 
                                                 strides=strides, 
                                                 padding="same", 
                                                 data_format=data_format, 
                                                 name=name + "/avgpool")(shortcut)
            strides = 1
        shortcut = ConvNormActBlock(filters=filters,
                                    kernel_size=1,
                                    strides=strides,
                                    data_format=data_format,
                                    trainable=trainable,
                                    kernel_initializer=kernel_initializer,
                                    normalization=None if preactivation else normalization,
                                    activation=None,
                                    dropblock=dropblock,
                                    gamma_zeros=init_gamma_zero,
                                    name=name + "/shortcut")(shortcut)
    
    if preactivation:
        return tf.keras.layers.Add(name=name + "/add")([x, shortcut])

    if se_ratio is not None and se_ratio > 0 and se_ratio <= 1:
        x = squeeze_excitation(
            x, in_filters=filters * expansion, se_ratio=se_ratio, data_format="channels_last", name=name + "/se")
    if drop_connect_rate:
        x = DropConnect(drop_connect_rate, name=name + "/drop_connect")(x)
    x = tf.keras.layers.Add(name=name + "/add")([x, shortcut])
    x = build_activation(**activation, name=name + "/" + activation["activation"])(x)
   
    return x


def bottleneck(inputs,
               filters,
               strides=1,
               dilation_rate=1,
               data_format="channels_last",
               kernel_initializer=tf.keras.initializers.VarianceScaling(),
               normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-3, axis=-1, trainable=True),
               init_gamma_zero=True,
               activation=dict(activation="relu"),
               trainable=True,
               dropblock=None,
               drop_connect_rate=None,
               se_ratio=None,
               use_conv_shortcut=True,
               preactivation=False,
               use_resnetd_shortcut=True,
               expansion=4,
               name=None):
    """A residual block.
        Args:
            filters: integer, filters of the bottleneck layer.
            convolution: The convolution type.
            strides: default 1, stride of the first layer.
            dilation_rate: default 1, dilation rate in 3x3 convolution.
            data_format: default channels_last,
            activation: the activation layer name.
            trainable: does this block is trainable.
            normalization: the normalization, e.g. "batch_norm", "group_norm" etc.
            dropblock: the arguments in DropBlock2D
            use_conv_shortcut: default True, use convolution shortcut if True,
                otherwise identity shortcut.
            name: string, block label.
    """
    if preactivation:
        x = build_normalization(name=name + "/norm1", **normalization)(inputs)
        x = build_activation(name=name + "/%s" % activation["activation"], **activation)(x)
    else:
        x = inputs
    
    shortcut = x
    x = ConvNormActBlock(filters=filters,
                         kernel_size=1,
                         strides=1,
                         trainable=trainable,
                         dropblock=dropblock,
                         kernel_initializer=kernel_initializer,
                         data_format=data_format,
                         normalization=normalization,
                         activation=activation,
                         name=name + "/conv1")(inputs)
    x = ConvNormActBlock(filters=filters,
                         kernel_size=3,
                         strides=strides,
                         dilation_rate=dilation_rate if strides == 1 else 1,
                         kernel_initializer=kernel_initializer,
                         trainable=trainable,
                         data_format=data_format,
                         normalization=normalization,
                         activation=activation,
                         name=name + "/conv2")(x)
    x = ConvNormActBlock(filters=expansion * filters,
                         kernel_size=1,
                         trainable=trainable,
                         data_format=data_format,
                         kernel_initializer=kernel_initializer,
                         normalization=None if preactivation else normalization,
                         activation=None,
                         name=name + "/conv3")(x)
   
    if use_conv_shortcut is True:
        if use_resnetd_shortcut and strides > 1:
            shortcut = tf.keras.layers.AvgPool2D(pool_size=strides,
                                                 strides=strides,
                                                 padding="same",
                                                 data_format=data_format,
                                                 name=name + "/shorcut/avgpool")(shortcut)
            strides = 1
        shortcut = ConvNormActBlock(filters=expansion * filters,
                                    kernel_size=1,
                                    strides=strides,
                                    data_format=data_format,
                                    kernel_initializer=kernel_initializer,
                                    trainable=trainable,
                                    dropblock=dropblock,
                                    normalization=None if preactivation else normalization,
                                    gamma_zeros=init_gamma_zero,
                                    activation=None,
                                    name=name + "/shortcut")(shortcut)
    if preactivation:
        return tf.keras.layers.Add(name=name + "/add")([x, shortcut])

    if se_ratio is not None and se_ratio > 0 and se_ratio <= 1:
        x = squeeze_excitation(
            x, in_filters=filters * expansion, se_ratio=se_ratio, data_format="channels_last", name=name + "/se")
    if drop_connect_rate:
        x = DropConnect(drop_connect_rate, name=name + "/drop_connect")(x)
    x = tf.keras.layers.Add(name=name + "/add")([x, shortcut])
    x = build_activation(**activation, name=name + "/" + activation["activation"])(x)
    
    return x


class ResNetRS(Extractor):
    def __init__(self,
                 name,
                 blocks,
                 block_fn,
                 kernel_initializer=tf.keras.initializers.VarianceScaling(),
                 normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                 init_gamma_zero=True,
                 activation=dict(activation="relu"),
                 strides=(2, 2, 2, 2, 2),
                 dilation_rates=(1, 1, 1, 1, 1),
                 frozen_stages=(-1,),
                 feat_dim=1000,
                 drop_rate=0.5,
                 dropblock=None,
                 drop_connect_rate=0.1,
                 se_ratio=None,
                 input_shape=(224, 224, 3),
                 input_tensor=None,
                 preactivation=False,
                 use_resnetd_stem=False,
                 skip_stem_max_pool=False,
                 replace_stem_max_pool=False,
                 use_resnetd_shortcut=True,
                 expansion=4,
                 **kwargs):
        super(ResNetRS, self).__init__(name=name,
                                       normalization=normalization,
                                       activation=activation,
                                       strides=strides,
                                       kernel_initializer=kernel_initializer,
                                       dilation_rates=dilation_rates,
                                       frozen_stages=frozen_stages,
                                       dropblock=dropblock,
                                       feat_dim=feat_dim,
                                       input_shape=input_shape,
                                       input_tensor=input_tensor,
                                       drop_rate=drop_rate,
                                       drop_connect_rate=drop_connect_rate,
                                       **kwargs)
        self.blocks = blocks
        self.block_fn = block_fn
        self.expansion = expansion
        self.se_ratio = se_ratio
        self.init_gamma_zero = init_gamma_zero
        self.preactivation = preactivation
        self.use_resnetd_stem = use_resnetd_stem
        self.skip_stem_max_pool = skip_stem_max_pool
        self.replace_stem_max_pool = replace_stem_max_pool
        self.use_resnetd_shortcut = use_resnetd_shortcut

        self.blockd_idx = 1
        self.num_blocks = sum(self.blocks)

    def build_model(self):
        # rgb_mean = self._rgb_mean
        # rgb_std = self._rgb_std

        def _norm(inp):
            rgb_mean = tf.constant([0.485 * 255, 0.456 * 255, 0.406 * 255], inp.dtype, [1, 1, 1, 3])
            rgb_std = tf.constant([0.229 * 255, 0.224 * 255, 0.225 * 255], inp.dtype, [1, 1, 1, 3])
            return (inp - rgb_mean) * (1. / rgb_std)

        x = tf.keras.layers.Lambda(function=_norm, name="norm_input")(self.img_input)

        if self.use_resnetd_stem:
            x = ConvNormActBlock(filters=32,
                                 kernel_size=3,
                                 strides=self.strides[0],
                                 dilation_rate=self.dilation_rates[0] if self.strides[0] == 1 else 1,
                                 trainable=1 not in self.frozen_stages,
                                 kernel_initializer=self.kernel_initializer,
                                 normalization=self.normalization,
                                 activation=self.activation,
                                 data_format=self.data_format,
                                 name="stem/conv1")(x)
            x = ConvNormActBlock(filters=32,
                                 kernel_size=3,
                                 strides=1,
                                 dilation_rate=self.dilation_rates[0],
                                 trainable=1 not in self.frozen_stages,
                                 kernel_initializer=self.kernel_initializer,
                                 normalization=self.normalization,
                                 activation=self.activation,
                                 data_format=self.data_format,
                                 name="stem/conv2")(x)
            x = ConvNormActBlock(filters=64,
                                 kernel_size=3,
                                 strides=1,
                                 dilation_rate=self.dilation_rates[0],
                                 trainable=1 not in self.frozen_stages,
                                 kernel_initializer=self.kernel_initializer,
                                 normalization=None if self.preactivation else self.normalization,
                                 activation=None if self.preactivation else self.activation,
                                 data_format=self.data_format,
                                 name="stem/conv3")(x)
        else:
            x = ConvNormActBlock(filters=64,
                                 kernel_size=7,
                                 strides=self.strides[0],
                                 dilation_rate=self.dilation_rates[0] if self.strides[0] == 1 else 1,
                                 trainable=1 not in self.frozen_stages,
                                 kernel_initializer=self.kernel_initializer,
                                 normalization=None if self.preactivation else self.normalization,
                                 activation=None if self.preactivation else self.activation,
                                 data_format=self.data_format,
                                 name="stem/conv1")(x)
        
        if not self.skip_stem_max_pool:
            if self.replace_stem_max_pool:
                x = ConvNormActBlock(filters=64,
                                     kernel_size=3,
                                     strides=self.strides[1],
                                     dilation_rate=1,
                                     trainable=1 not in self.frozen_stages,
                                     kernel_initializer=self.kernel_initializer,
                                     normalization=None if self.preactivation else self.normalization,
                                     activation=None if self.preactivation else self.activation,
                                     data_format=self.data_format,
                                     name="stem/maxpool")(x)
            else:
                x = tf.keras.layers.MaxPool2D(3, self.strides[1], "same", self.data_format, name="stem/maxpool")(x)

        self.in_filters = 64 
        trainable = 2 not in self.frozen_stages
        x = self.stack(x, 64, self.strides[1] if self.skip_stem_max_pool else 1, self.dilation_rates[1], trainable, self.blocks[0], "layer1")
        trainable = 3 not in self.frozen_stages
        x = self.stack(x, 128, self.strides[2], self.dilation_rates[2], trainable, self.blocks[1], "layer2")
        trainable = 4 not in self.frozen_stages
        x = self.stack(x, 256, self.strides[3], self.dilation_rates[3], trainable, self.blocks[2], "layer3")
        trainable = 5 not in self.frozen_stages
        x = self.stack(x, 512, self.strides[4], self.dilation_rates[4], trainable, self.blocks[3], "layer4")
        if self.preactivation:
            x = build_normalization(**self.normalization)(x)
            x = build_activation(**self.activation)(x)
        x = tf.keras.layers.GlobalAvgPool2D(data_format=self.data_format)(x)
        x = tf.keras.layers.Dropout(rate=self.drop_rate)(x)
        x = tf.keras.layers.Dense(self.feat_dim, name="logits")(x)
        
        model = tf.keras.Model(inputs=self.img_input, outputs=x, name=self.name)

        return model

    def stack(self, x, filters, strides, dilation_rate, trainable, blocks, name=None):
        use_conv_shortcut = False
        if strides != 1 or self.in_filters != filters * self.expansion:
            use_conv_shortcut = True
        x = self.block_fn(inputs=x,
                          filters=filters,
                          strides=strides,
                          dilation_rate=dilation_rate,
                          kernel_initializer=self.kernel_initializer,
                          normalization=self.normalization,
                          init_gamma_zero=self.init_gamma_zero,
                          activation=self.activation,
                          trainable=trainable,
                          dropblock=self.dropblock,
                          drop_connect_rate=get_drop_connect_rate(self.drop_connect_rate, self.blockd_idx, self.num_blocks),
                          use_conv_shortcut=use_conv_shortcut,
                          se_ratio=self.se_ratio,
                          name=name + "/0")
        self.blockd_idx += 1
        for i in range(1, blocks):
            x = self.block_fn(inputs=x,
                              filters=filters,
                              strides=1,
                              dilation_rate=dilation_rate,
                              kernel_initializer=self.kernel_initializer,
                              normalization=self.normalization,
                              activation=self.activation,
                              trainable=trainable,
                              init_gamma_zero=self.init_gamma_zero,
                              dropblock=self.dropblock,
                              use_conv_shortcut=False,
                              se_ratio=self.se_ratio,
                              drop_connect_rate=get_drop_connect_rate(self.drop_connect_rate, self.blockd_idx, self.num_blocks),
                              name=name + "/%d" % i)
            self.blockd_idx += 1
        return x


@EXTRACTORS.register("ResNetRS50")
def ResNetRS50(normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
               activation=dict(activation="relu"),
               strides=(2, 2, 2, 2, 2),
               dilation_rates=(1, 1, 1, 1, 1),
               frozen_stages=(-1, ),
               dropblock=None,
               feat_dim=512,
               drop_rate=0.25,
               drop_connect_rate=None,
               input_shape=(160, 160, 3),
               input_tensor=None,
               **kwargs):

    return ResNetRS(name="resnet_rs_50",
                    blocks=[3, 4, 6, 3],
                    block_fn=bottleneck,
                    normalization=normalization,
                    activation=activation,
                    strides=strides,
                    dilation_rates=dilation_rates,
                    frozen_stages=frozen_stages,
                    dropblock=dropblock,
                    feat_dim=feat_dim,
                    input_shape=input_shape,
                    expansion=4,
                    input_tensor=input_tensor,
                    drop_rate=drop_rate,
                    drop_connect_rate=drop_connect_rate,
                    use_resnetd_stem=True,
                    replace_stem_max_pool=True,
                    skip_stem_max_pool=False,
                    use_resnetd_shortcut=True,
                    se_ratio=0.25,
                    **kwargs).build_model()


@EXTRACTORS.register("ResNetRS101")
def ResNetRS101(normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                activation=dict(activation="relu"),
                strides=(2, 2, 2, 2, 2),
                dilation_rates=(1, 1, 1, 1, 1),
                frozen_stages=(-1, ),
                dropblock=None,
                feat_dim=512,
                drop_rate=0.25,
                drop_connect_rate=None,
                input_shape=(160, 160, 3),
                input_tensor=None,
                **kwargs):

    return ResNetRS(name="resnet_rs_101",
                    blocks=[3, 4, 23, 3],
                    block_fn=bottleneck,
                    normalization=normalization,
                    activation=activation,
                    strides=strides,
                    dilation_rates=dilation_rates,
                    frozen_stages=frozen_stages,
                    dropblock=dropblock,
                    feat_dim=feat_dim,
                    input_shape=input_shape,
                    expansion=4,
                    input_tensor=input_tensor,
                    drop_rate=drop_rate,
                    drop_connect_rate=drop_connect_rate,
                    use_resnetd_stem=True,
                    replace_stem_max_pool=True,
                    skip_stem_max_pool=False,
                    use_resnetd_shortcut=True,
                    se_ratio=0.25,
                    **kwargs).build_model()


@EXTRACTORS.register("ResNetRS152")
def ResNetRS152(normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                activation=dict(activation="relu"),
                strides=(2, 2, 2, 2, 2),
                dilation_rates=(1, 1, 1, 1, 1),
                frozen_stages=(-1, ),
                dropblock=None,
                feat_dim=512,
                drop_rate=0.25,
                drop_connect_rate=None,
                input_shape=(160, 160, 3),
                input_tensor=None,
                **kwargs):

    return ResNetRS(name="resnet_rs_152",
                    blocks=[3, 8, 36, 3],
                    block_fn=bottleneck,
                    normalization=normalization,
                    activation=activation,
                    strides=strides,
                    dilation_rates=dilation_rates,
                    frozen_stages=frozen_stages,
                    dropblock=dropblock,
                    feat_dim=feat_dim,
                    input_shape=input_shape,
                    expansion=4,
                    input_tensor=input_tensor,
                    drop_rate=drop_rate,
                    drop_connect_rate=drop_connect_rate,
                    use_resnetd_stem=True,
                    replace_stem_max_pool=True,
                    skip_stem_max_pool=False,
                    use_resnetd_shortcut=True,
                    se_ratio=0.25,
                    **kwargs).build_model()


@EXTRACTORS.register("ResNetRS200")
def ResNetRS200(normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                activation=dict(activation="relu"),
                strides=(2, 2, 2, 2, 2),
                dilation_rates=(1, 1, 1, 1, 1),
                frozen_stages=(-1, ),
                dropblock=None,
                feat_dim=512,
                drop_rate=0.25,
                drop_connect_rate=0.1,
                input_shape=(224, 224, 3),
                input_tensor=None,
                **kwargs):

    return ResNetRS(name="resnet_rs_200",
                    blocks=[3, 24, 36, 3],
                    block_fn=bottleneck,
                    normalization=normalization,
                    activation=activation,
                    strides=strides,
                    dilation_rates=dilation_rates,
                    frozen_stages=frozen_stages,
                    dropblock=dropblock,
                    feat_dim=feat_dim,
                    input_shape=input_shape,
                    expansion=4,
                    input_tensor=input_tensor,
                    drop_rate=drop_rate,
                    drop_connect_rate=drop_connect_rate,
                    use_resnetd_stem=True,
                    replace_stem_max_pool=True,
                    skip_stem_max_pool=False,
                    use_resnetd_shortcut=True,
                    se_ratio=0.25,
                    **kwargs).build_model()


@EXTRACTORS.register("ResNetRS270")
def ResNetRS270(normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                activation=dict(activation="relu"),
                strides=(2, 2, 2, 2, 2),
                dilation_rates=(1, 1, 1, 1, 1),
                frozen_stages=(-1, ),
                dropblock=None,
                feat_dim=512,
                drop_rate=0.25,
                drop_connect_rate=0.1,
                input_shape=(224, 224, 3),
                input_tensor=None,
                **kwargs):

    return ResNetRS(name="resnet_rs_270",
                    blocks=[4, 29, 53, 4],
                    block_fn=bottleneck,
                    normalization=normalization,
                    activation=activation,
                    strides=strides,
                    dilation_rates=dilation_rates,
                    frozen_stages=frozen_stages,
                    dropblock=dropblock,
                    feat_dim=feat_dim,
                    input_shape=input_shape,
                    expansion=4,
                    input_tensor=input_tensor,
                    drop_rate=drop_rate,
                    drop_connect_rate=drop_connect_rate,
                    use_resnetd_stem=True,
                    replace_stem_max_pool=True,
                    skip_stem_max_pool=False,
                    use_resnetd_shortcut=True,
                    se_ratio=0.25,
                    **kwargs).build_model()


@EXTRACTORS.register("ResNetRS350")
def ResNetRS350(normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                activation=dict(activation="relu"),
                strides=(2, 2, 2, 2, 2),
                dilation_rates=(1, 1, 1, 1, 1),
                frozen_stages=(-1, ),
                dropblock=None,
                feat_dim=512,
                drop_rate=0.25,
                drop_connect_rate=0.1,
                input_shape=(224, 224, 3),
                input_tensor=None,
                **kwargs):

    return ResNetRS(name="resnet_rs_350",
                    blocks=[4, 36, 72, 4],
                    block_fn=bottleneck,
                    normalization=normalization,
                    activation=activation,
                    strides=strides,
                    dilation_rates=dilation_rates,
                    frozen_stages=frozen_stages,
                    dropblock=dropblock,
                    feat_dim=feat_dim,
                    input_shape=input_shape,
                    expansion=4,
                    input_tensor=input_tensor,
                    drop_rate=drop_rate,
                    drop_connect_rate=drop_connect_rate,
                    use_resnetd_stem=True,
                    replace_stem_max_pool=True,
                    skip_stem_max_pool=False,
                    use_resnetd_shortcut=True,
                    se_ratio=0.25,
                    **kwargs).build_model()


@EXTRACTORS.register("ResNetRS420")
def ResNetRS420(normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                activation=dict(activation="relu"),
                strides=(2, 2, 2, 2, 2),
                dilation_rates=(1, 1, 1, 1, 1),
                frozen_stages=(-1, ),
                dropblock=None,
                feat_dim=512,
                drop_rate=0.4,
                drop_connect_rate=0.1,
                input_shape=(224, 224, 3),
                input_tensor=None,
                **kwargs):

    return ResNetRS(name="resnet_rs_420",
                    blocks=[4, 44, 87, 4],
                    block_fn=bottleneck,
                    normalization=normalization,
                    activation=activation,
                    strides=strides,
                    dilation_rates=dilation_rates,
                    frozen_stages=frozen_stages,
                    dropblock=dropblock,
                    feat_dim=feat_dim,
                    input_shape=input_shape,
                    expansion=4,
                    input_tensor=input_tensor,
                    drop_rate=drop_rate,
                    drop_connect_rate=drop_connect_rate,
                    use_resnetd_stem=True,
                    replace_stem_max_pool=True,
                    skip_stem_max_pool=False,
                    use_resnetd_shortcut=True,
                    se_ratio=0.25,
                    **kwargs).build_model()


def _get_weight_name_map(blocks):
    name_map = {
        "stem/conv1/conv2d/kernel:0": "conv2d/kernel",
        "stem/conv1/batch_norm/gamma:0": "batch_normalization/gamma",
        "stem/conv1/batch_norm/beta:0": "batch_normalization/beta",
        "stem/conv1/batch_norm/moving_mean:0": "batch_normalization/moving_mean",
        "stem/conv1/batch_norm/moving_variance:0": "batch_normalization/moving_variance",
        "stem/conv2/conv2d/kernel:0": "conv2d_1/kernel",
        "stem/conv2/batch_norm/gamma:0": "batch_normalization_1/gamma",
        "stem/conv2/batch_norm/beta:0": "batch_normalization_1/beta",
        "stem/conv2/batch_norm/moving_mean:0": "batch_normalization_1/moving_mean",
        "stem/conv2/batch_norm/moving_variance:0": "batch_normalization_1/moving_variance",
        "stem/conv3/conv2d/kernel:0": "conv2d_2/kernel",
        "stem/conv3/batch_norm/gamma:0": "batch_normalization_2/gamma",
        "stem/conv3/batch_norm/beta:0": "batch_normalization_2/beta",
        "stem/conv3/batch_norm/moving_mean:0": "batch_normalization_2/moving_mean",
        "stem/conv3/batch_norm/moving_variance:0": "batch_normalization_2/moving_variance",
        "stem/maxpool/conv2d/kernel:0": "conv2d_3/kernel",
        "stem/maxpool/batch_norm/gamma:0": "batch_normalization_3/gamma",
        "stem/maxpool/batch_norm/beta:0": "batch_normalization_3/beta",
        "stem/maxpool/batch_norm/moving_mean:0": "batch_normalization_3/moving_mean",
        "stem/maxpool/batch_norm/moving_variance:0": "batch_normalization_3/moving_variance",
    }

    cnt = 4
    bn_cnt = 4
    for i in range(1, 5):
        for j in range(blocks[i - 1]):
            if j == 0:
                m = {
                    "layer%d/0/shortcut/conv2d/kernel:0" % i: "conv2d_%d/kernel" % cnt,
                    "layer%d/0/shortcut/batch_norm/gamma:0" % i: "batch_normalization_%d/gamma" % bn_cnt,
                    "layer%d/0/shortcut/batch_norm/beta:0" % i: "batch_normalization_%d/beta" % bn_cnt,
                    "layer%d/0/shortcut/batch_norm/moving_mean:0" % i: "batch_normalization_%d/moving_mean" % bn_cnt,
                    "layer%d/0/shortcut/batch_norm/moving_variance:0" % i: "batch_normalization_%d/moving_variance" % bn_cnt
                }
                name_map.update(m)
                cnt += 1
                bn_cnt += 1
            for k in range(1, 4):
                n1 = "layer%d/%d/conv%d" % (i, j, k)
                m = {
                    "%s/conv2d/kernel:0" % n1: "conv2d_%d/kernel" % cnt, 
                    "%s/batch_norm/gamma:0" % n1: "batch_normalization_%d/gamma" % bn_cnt,
                    "%s/batch_norm/beta:0" % n1: "batch_normalization_%d/beta" % bn_cnt,
                    "%s/batch_norm/moving_mean:0" % n1: "batch_normalization_%d/moving_mean" % bn_cnt,
                    "%s/batch_norm/moving_variance:0" % n1: "batch_normalization_%d/moving_variance" % bn_cnt,
                }
                cnt += 1
                bn_cnt += 1
                name_map.update(m)
           
            m = {
                "layer%d/%d/se/squeeze/kernel:0" % (i, j): "conv2d_%d/kernel" % cnt,
                "layer%d/%d/se/squeeze/bias:0" % (i, j): "conv2d_%d/bias" % cnt,
                "layer%d/%d/se/excitation/kernel:0" % (i, j): "conv2d_%d/kernel" % (cnt + 1),
                "layer%d/%d/se/excitation/bias:0" % (i, j): "conv2d_%d/bias" % (cnt + 1),
            }
            name_map.update(m)
            cnt += 2
    
    name_map["logits/kernel:0"] = "dense/kernel"
    name_map["logits/bias:0"] = "dense/bias"

    return name_map


def _get_weights_from_pretrained(model, pretrained_weights_path, blocks):
    ckpt = tf.train.latest_checkpoint(pretrained_weights_path)
    for w in tf.train.list_variables(ckpt):
        if "ExponentialMovingAverage" not in w[0]:
            use_exponential_moving_average = True
        # if "ExponentialMovingAverage" not in w[0] and "Momentum" not in w[0]:
        #     print(w[0], w[1])
    name_map = _get_weight_name_map(blocks)
    # for k, v in name_map.items():
    #     print(k, v,)
    for w in model.weights:
        name = name_map[w.name]
        # print(name, w.shape.as_list())
        if use_exponential_moving_average:
            name += "/ExponentialMovingAverage"
        try:
            pw = tf.train.load_variable(ckpt, name)
            w.assign(pw)
        except Exception as e:
            print(str(e))
            pass


if __name__ == "__main__":
    # from ..common import fuse
    model_params = {
      "resnet-rs-18": [2, 2, 2, 2],
      "resnet-rs-34": [3, 4, 6, 3],
      "resnet-rs-50": [3, 4, 6, 3],
      "resnet-rs-101": [3, 4, 23, 3],
      "resnet-rs-152": [3, 8, 36, 3],
      "resnet-rs-200": [3, 24, 36, 3],
      "resnet-rs-270": [4, 29, 53, 4],
      "resnet-rs-350": [4, 36, 72, 4],
      "resnet-rs-420": [4, 44, 87, 4]
  }
    name = "resnet-rs-420"
    block_fn = bottleneck
    blocks = model_params[name]
    sz = 320
    name = "%s-i%d" % (name, sz)
    resnet = ResNetRS420(feat_dim=1000, input_shape=(sz, sz, 3))
    # resnet.summary()
    _get_weights_from_pretrained(resnet, "/Users/bailang/Downloads/%s" % name, blocks)
    
    # fuse(model, block_fn)

    with tf.io.gfile.GFile("/Users/bailang/Documents/pandas.jpg", "rb") as gf:
        images = tf.image.decode_jpeg(gf.read())

    images = tf.image.resize(images, (sz, sz))
    images = tf.expand_dims(images, axis=0)
    lbl = resnet(images, training=False)
    top5prob, top5class = tf.nn.top_k(tf.squeeze(tf.nn.softmax(lbl, -1), axis=0), k=5)
    print("prob:", top5prob.numpy())
    print("class:", top5class.numpy())
    
    # tf.saved_model.save(resnet, "./resnet50")
    resnet.save_weights("/Users/bailang/Downloads/pretrained_weights/%s.h5" % name)
    resnet.save_weights("/Users/bailang/Downloads/pretrained_weights/%s/model.ckpt" % name)
