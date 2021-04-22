import math
import tensorflow as tf
from .backbone import Backbone
from ..builder import BACKBONES
from ..common import ConvNormActBlock
from core.layers import build_activation


def bottleneckx(inputs,
                filters,
                cardinality,
                bottleneck_width,
                strides=1,
                dilation_rate=1,
                use_se=False,
                normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                activation=dict(activation="relu"),
                trainable=True,
                downsample=False,
                dropblock=None,
                avg_down=False,
                last_gamma=False,
                data_format="channels_last",
                expansion=4,
                name=None):
    d = int(math.floor(filters * (bottleneck_width / 64)))
    group_width = cardinality * d

    x = ConvNormActBlock(filters=group_width,
                         kernel_size=1,
                         strides=1,
                         trainable=trainable,
                         data_format=data_format,
                         dropblock=dropblock,
                         normalization=normalization,
                         activation=activation,
                         name=name + "/conv1")(inputs)
    x = ConvNormActBlock(filters=group_width,
                         kernel_size=(3, 3),
                         strides=strides,
                         groups=cardinality,
                         dilation_rate=1 if strides == 1 else dilation_rate,
                         data_format=data_format,
                         trainable=trainable,
                         normalization=normalization,
                         activation=activation,
                         dropblock=dropblock,
                         name=name + "/conv2")(x)
    x = ConvNormActBlock(filters=filters * expansion,
                         kernel_size=1,
                         trainable=trainable,
                         data_format=data_format,
                         normalization=normalization,
                         activation=None,
                         dropblock=dropblock,
                         name=name + "/conv3")(x)
    
    if use_se:
        pool_axis = [1, 2] if data_format == "channels_last" else [2, 3]
        se = tf.keras.layers.Lambda(lambda inp: tf.reduce_mean(inp, pool_axis, keepdims=True), 
                                    name=name + "/se/pool")(x)
        se = tf.keras.layers.Conv2D(filters=filters // 4,
                                    kernel_size=(1, 1),
                                    data_format=data_format,
                                    name=name + "/conv4")(se)

        se = tf.keras.layers.Conv2D(filters=filters * expansion,
                                    kernel_size=(1, 1),
                                    data_format=data_format,
                                    name=name + "/conv5")(se)
        se = tf.keras.layers.Activation("sigmoid", name=name + "/sigmoid")(se)
        x = tf.keras.layers.Multiply(name=name + "/multiply")([x, se])

    shortcut = inputs
    if downsample:
        if avg_down:
            shortcut = tf.keras.layers.AvgPool2D(pool_size=strides, 
                                                 strides=strides, 
                                                 padding="same", 
                                                 data_format=data_format, 
                                                 name=name + "/shorcut/avg_pool")(shortcut)
            shortcut = ConvNormActBlock(kernel_size=1, 
                                        filters=filters * expansion,
                                        trainable=trainable,
                                        data_format=data_format,
                                        normalization=normalization,
                                        activation=None,
                                        dropblock=dropblock,
                                        name=name + "/shortcut")(shortcut)
        else:
            shortcut = ConvNormActBlock(kernel_size=1, 
                                        strides=strides,
                                        filters=filters * expansion,
                                        trainable=trainable,
                                        data_format=data_format,
                                        normalization=normalization,
                                        activation=None,
                                        dropblock=dropblock,
                                        name=name + "/shortcut")(shortcut)
    x = tf.keras.layers.Add(name=name + "/add")([x, shortcut])
    x = build_activation(**activation, name=name + "/" + activation["activation"])(x)
    
    return x
            

class ResNeXt(Backbone):
    def __init__(self, 
                 name, 
                 deep_stem=False,
                 block_fn=bottleneckx,
                 num_blocks=(3, 4, 6, 3),
                 stem_filters=32,
                 cardinality=32,
                 bottleneck_width=64,
                 use_se=False,
                 normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                 activation=dict(activation="relu"),
                 output_indices=(3, 4), 
                 strides=(2, 2, 2, 2, 2), 
                 dilation_rates=(1, 1, 1, 1, 1), 
                 frozen_stages=(-1, ), 
                 input_shape=None, 
                 input_tensor=None, 
                 dropblock=None, 
                 last_gamma=False,
                 avg_down=False,
                 num_classes=1000,
                 drop_rate=0.5,
                 **kwargs):
        super(ResNeXt, self).__init__(name,
                                      normalization=normalization, 
                                      activation=activation, 
                                      output_indices=output_indices, 
                                      strides=strides, 
                                      dilation_rates=dilation_rates, 
                                      frozen_stages=frozen_stages, 
                                      input_shape=input_shape, 
                                      input_tensor=input_tensor,
                                      dropblock=dropblock, 
                                      num_classes=num_classes,
                                      drop_rate=drop_rate,
                                      **kwargs)
        self.deep_stem = deep_stem
        self.block_fn = block_fn
        self.num_blocks = num_blocks
        self.stem_filters = stem_filters
        self.last_gamma = last_gamma
        self.avg_down = avg_down
        self.cardinality = cardinality
        self.bottleneck_width = bottleneck_width
        self.use_se = use_se

    def build_model(self):
        def _norm(inp):
            mean = tf.constant([0.485, 0.456, 0.406], inp.dtype, [1, 1, 1, 3]) * 255.
            std = 1. / (tf.constant([0.229, 0.224, 0.225], inp.dtype, [1, 1, 1, 3]) * 255.)
            return (inp - mean) * std  

        x = tf.keras.layers.Lambda(_norm, name="norm_input")(self.img_input)
        if not self.deep_stem:
            x = ConvNormActBlock(filters=64,
                                 kernel_size=(7, 7),
                                 strides=self.strides[0],
                                 dilation_rate=self.dilation_rates[0],
                                 trainable=1 not in self.frozen_stages,
                                 kernel_initializer="he_normal",
                                 normalization=self.normalization,
                                 data_format=self.data_format,
                                 name="stem/conv1")(x)
        else:
            x = ConvNormActBlock(filters=self.stem_filters,
                                 kernel_size=(3, 3),
                                 strides=self.strides[0],
                                 dilation_rate=self.dilation_rates[0],
                                 trainable=1 not in self.frozen_stages,
                                 kernel_initializer="he_normal",
                                 normalization=self.normalization,
                                 data_format=self.data_format,
                                 name="stem/conv1")(x)
            x = ConvNormActBlock(filters=self.stem_filters,
                                 kernel_size=(3, 3),
                                 strides=(1, 1),
                                 dilation_rate=self.dilation_rates[0],
                                 trainable=1 not in self.frozen_stages,
                                 kernel_initializer="he_normal",
                                 normalization=self.normalization,
                                 data_format=self.data_format,
                                 name="stem/conv2")(x)
            x = ConvNormActBlock(filters=self.stem_filters * 2,
                                 kernel_size=(7, 7),
                                 strides=(1, 1),
                                 dilation_rate=self.dilation_rates[0],
                                 trainable=1 not in self.frozen_stages,
                                 kernel_initializer="he_normal",
                                 normalization=self.normalization,
                                 data_format=self.data_format,
                                 name="stem/conv3")(x)
        
        outputs = [x]
        x = tf.keras.layers.ZeroPadding2D(data_format=self.data_format, padding=((1, 1), (1, 1)))(x)
        x = tf.keras.layers.MaxPool2D((3, 3), self.strides[1], "valid", self.data_format)(x)
        x = self._make_stage(x, 64, self.num_blocks[0], 1, self.dilation_rates[1], 2 not in self.frozen_stages, "layer1")
        outputs.append(x)
        x = self._make_stage(x, 128, self.num_blocks[1], self.strides[2], self.dilation_rates[2], 3 not in self.frozen_stages, "layer2")
        outputs.append(x)
        x = self._make_stage(x, 256, self.num_blocks[2], self.strides[3], self.dilation_rates[3], 4 not in self.frozen_stages, "layer3")
        outputs.append(x)
        x = self._make_stage(x, 512, self.num_blocks[3], self.strides[4], self.dilation_rates[4], 5 not in self.frozen_stages, "layer4")
        outputs.append(x)
        
        if -1 not in self.output_indices:
            outputs = [outputs[i-1] for i in self.output_indices]
        else:
            x = tf.keras.layers.GlobalAvgPool2D(data_format=self.data_format)(x)
            if self.drop_rate and self.drop_rate > 0.:
                x = tf.keras.layers.Dropout(rate=self.drop_rate, name="dropout")(x)
        
            outputs = tf.keras.layers.Dense(units=self.num_classes, name="logits")(x)

        return tf.keras.Model(inputs=self.img_input, outputs=outputs, name=self.name)

    def _make_stage(self, x, filters, num_blocks, strides=1, dilation_rate=1, trainable=True, name=None):
        block_index = 0
        x = self.block_fn(x,
                          cardinality=self.cardinality,
                          use_se=self.use_se,
                          bottleneck_width=self.bottleneck_width,
                          filters=filters,
                          strides=strides,
                          dilation_rate=dilation_rate,
                          normalization=self.normalization,
                          activation=self.activation,
                          trainable=trainable,
                          downsample=True,
                          dropblock=self.dropblock,
                          last_gamma=self.last_gamma,
                          avg_down=self.avg_down,
                          data_format=self.data_format,
                          name=name + "/0")
        block_index = block_index + 6 if self.use_se else block_index + 4
        for i in range(1, num_blocks):
            x = self.block_fn(x, 
                              cardinality=self.cardinality,
                              use_se=self.use_se,
                              bottleneck_width=self.bottleneck_width,
                              filters=filters,
                              strides=1,
                              dilation_rate=dilation_rate,
                              normalization=self.normalization,
                              activation=self.activation,
                              trainable=trainable,
                              downsample=False,
                              dropblock=self.dropblock,
                              last_gamma=self.last_gamma,
                              avg_down=self.avg_down,
                              data_format=self.data_format,
                              name=name + "/%d" % i)
            n = 5 if self.use_se else 3
            block_index += n
        return x
    

@BACKBONES.register("ResNeXt50_32X4D")
def ResNeXt50_32X4D(normalization=dict(normalization ='batch_norm', momentum =0.9, epsilon =1e-05, axis = -1, trainable =True), 
                    activation=dict(activation ='relu'), 
                    output_indices=(3, 4), 
                    strides=(2, 2, 2, 2, 2), 
                    dilation_rates=(1, 1, 1, 1, 1), 
                    frozen_stages=(-1, ), 
                    input_shape=None, 
                    input_tensor=None, 
                    dropblock=None, 
                    last_gamma=False, 
                    num_classes=1000, 
                    drop_rate=0.5,
                    **kwargs):
    return ResNeXt("resnext50_32x4d", 
                   deep_stem=False, 
                   block_fn=bottleneckx, 
                   num_blocks=[3, 4, 6, 3], 
                   stem_filters=32, 
                   cardinality=32,
                   use_se=False,
                   bottleneck_width=4,
                   normalization=normalization, 
                   activation=activation, 
                   output_indices=output_indices, 
                   strides=strides, 
                   dilation_rates=dilation_rates, 
                   frozen_stages=frozen_stages, 
                   input_shape=input_shape, 
                   input_tensor=input_tensor, 
                   dropblock=dropblock, 
                   last_gamma=last_gamma, 
                   avg_down=False, 
                   num_classes=num_classes, 
                   drop_rate=drop_rate,
                   **kwargs).build_model()


@BACKBONES.register("ResNeXt101_32X4D")
def ResNeXt101_32X4D(normalization=dict(normalization ='batch_norm', momentum =0.9, epsilon =1e-05, axis = -1, trainable =True), 
                     activation=dict(activation ='relu'), 
                     output_indices=(3, 4), 
                     strides=(2, 2, 2, 2, 2), 
                     dilation_rates=(1, 1, 1, 1, 1), 
                     frozen_stages=(-1, ), 
                     input_shape=None, 
                     input_tensor=None, 
                     dropblock=None, 
                     last_gamma=False, 
                     num_classes=1000, 
                     drop_rate=0.5,
                     **kwargs):
    return ResNeXt("resnext101_32x4d", 
                   deep_stem=False, 
                   block_fn=bottleneckx, 
                   num_blocks=[3, 4, 23, 3], 
                   stem_filters=32, 
                   cardinality=32,
                   use_se=False,
                   bottleneck_width=4,
                   normalization=normalization, 
                   activation=activation, 
                   output_indices=output_indices, 
                   strides=strides, 
                   dilation_rates=dilation_rates, 
                   frozen_stages=frozen_stages, 
                   input_shape=input_shape, 
                   input_tensor=input_tensor, 
                   dropblock=dropblock, 
                   last_gamma=last_gamma, 
                   avg_down=False, 
                   num_classes=num_classes,                     
                   drop_rate=drop_rate,
                   **kwargs).build_model()


@BACKBONES.register("ResNeXt101_64X4D")
def ResNeXt101_64X4D(normalization=dict(normalization ='batch_norm', momentum =0.9, epsilon =1e-05, axis = -1, trainable =True), 
                     activation=dict(activation ='relu'), 
                     output_indices=(3, 4), 
                     strides=(2, 2, 2, 2, 2), 
                     dilation_rates=(1, 1, 1, 1, 1), 
                     frozen_stages=(-1, ), 
                     input_shape=None, 
                     input_tensor=None, 
                     dropblock=None, 
                     last_gamma=False, 
                     num_classes=1000, 
                     drop_rate=0.5,
                     **kwargs):
    return ResNeXt("resnext101_64x4d", 
                   deep_stem=False, 
                   block_fn=bottleneckx, 
                   num_blocks=[3, 4, 23, 3], 
                   stem_filters=32, 
                   cardinality=64,
                   use_se=False,
                   bottleneck_width=4,
                   normalization=normalization, 
                   activation=activation, 
                   output_indices=output_indices, 
                   strides=strides, 
                   dilation_rates=dilation_rates, 
                   frozen_stages=frozen_stages, 
                   input_shape=input_shape, 
                   input_tensor=input_tensor, 
                   dropblock=dropblock, 
                   last_gamma=last_gamma, 
                   avg_down=False, 
                   num_classes=num_classes, 
                   drop_rate=drop_rate,
                   **kwargs).build_model()


@BACKBONES.register("ResNeXt101B_64X4D")
def ResNeXt101B_64X4D(normalization=dict(normalization ='batch_norm', momentum =0.9, epsilon =1e-05, axis = -1, trainable =True), 
                      activation=dict(activation ='relu'), 
                      output_indices=(3, 4), 
                      strides=(2, 2, 2, 2, 2), 
                      dilation_rates=(1, 1, 1, 1, 1), 
                      frozen_stages=(-1, ), 
                      input_shape=None, 
                      input_tensor=None, 
                      dropblock=None, 
                      last_gamma=False, 
                      num_classes=1000, 
                      drop_rate=0.5,
                      **kwargs):
    return ResNeXt("resnext101b_64x4d", 
                   deep_stem=True, 
                   block_fn=bottleneckx, 
                   num_blocks=[3, 4, 23, 3], 
                   stem_filters=32, 
                   cardinality=64,
                   use_se=False,
                   bottleneck_width=4,
                   normalization=normalization, 
                   activation=activation, 
                   output_indices=output_indices, 
                   strides=strides, 
                   dilation_rates=dilation_rates, 
                   frozen_stages=frozen_stages, 
                   input_shape=input_shape, 
                   input_tensor=input_tensor, 
                   dropblock=dropblock, 
                   last_gamma=last_gamma, 
                   avg_down=True, 
                   num_classes=num_classes, 
                   drop_rate=drop_rate,
                   **kwargs).build_model()


@BACKBONES.register("SEResNeXt50_32X4D")
def SEResNeXt50_32X4D(normalization=dict(normalization ='batch_norm', momentum =0.9, epsilon =1e-05, axis = -1, trainable =True), 
                      activation=dict(activation ='relu'), 
                      output_indices=(3, 4), 
                      strides=(2, 2, 2, 2, 2), 
                      dilation_rates=(1, 1, 1, 1, 1), 
                      frozen_stages=(-1, ), 
                      input_shape=None, 
                      input_tensor=None, 
                      dropblock=None, 
                      last_gamma=False, 
                      num_classes=1000, 
                      drop_rate=0.5,
                      **kwargs):
    return ResNeXt("se_resnext50_32x4d", 
                   deep_stem=False, 
                   block_fn=bottleneckx, 
                   num_blocks=[3, 4, 6, 3], 
                   stem_filters=32, 
                   cardinality=32,
                   use_se=True,
                   bottleneck_width=4,
                   normalization=normalization, 
                   activation=activation, 
                   output_indices=output_indices, 
                   strides=strides, 
                   dilation_rates=dilation_rates, 
                   frozen_stages=frozen_stages, 
                   input_shape=input_shape, 
                   input_tensor=input_tensor, 
                   dropblock=dropblock, 
                   last_gamma=last_gamma, 
                   avg_down=False, 
                   num_classes=num_classes, 
                   drop_rate=drop_rate,
                   **kwargs).build_model()


@BACKBONES.register("SEResNeXt101_32X4D")
def SEResNeXt101_32X4D(normalization=dict(normalization ='batch_norm', momentum =0.9, epsilon =1e-05, axis = -1, trainable =True), 
                       activation=dict(activation ='relu'), 
                       output_indices=(3, 4), 
                       strides=(2, 2, 2, 2, 2), 
                       dilation_rates=(1, 1, 1, 1, 1), 
                       frozen_stages=(-1, ), 
                       input_shape=None, 
                       input_tensor=None, 
                       dropblock=None, 
                       last_gamma=False, 
                       num_classes=1000, 
                       drop_rate=0.5,
                       **kwargs):
    return ResNeXt("se_resnext101_32x4d", 
                   deep_stem=False, 
                   block_fn=bottleneckx, 
                   num_blocks=[3, 4, 23, 3], 
                   stem_filters=32, 
                   cardinality=32,
                   use_se=True,
                   bottleneck_width=4,
                   normalization=normalization, 
                   activation=activation, 
                   output_indices=output_indices, 
                   strides=strides, 
                   dilation_rates=dilation_rates, 
                   frozen_stages=frozen_stages, 
                   input_shape=input_shape, 
                   input_tensor=input_tensor, 
                   dropblock=dropblock, 
                   last_gamma=last_gamma, 
                   avg_down=False, 
                   num_classes=num_classes, 
                   drop_rate=drop_rate,
                   **kwargs).build_model()


@BACKBONES.register("SEResNeXt101_64X4D")
def SEResNeXt101_64X4D(normalization=dict(normalization ='batch_norm', momentum =0.9, epsilon =1e-05, axis = -1, trainable =True), 
                       activation=dict(activation ='relu'), 
                       output_indices=(3, 4), 
                       strides=(2, 2, 2, 2, 2), 
                       dilation_rates=(1, 1, 1, 1, 1), 
                       frozen_stages=(-1, ), 
                       input_shape=None, 
                       input_tensor=None, 
                       dropblock=None, 
                       last_gamma=False, 
                       num_classes=1000, 
                       drop_rate=0.5,
                       **kwargs):
    return ResNeXt("se_resnext101_64x4d", 
                   deep_stem=False, 
                   block_fn=bottleneckx, 
                   num_blocks=[3, 4, 23, 3], 
                   stem_filters=32, 
                   cardinality=64,
                   use_se=True,
                   bottleneck_width=4,
                   normalization=normalization, 
                   activation=activation, 
                   output_indices=output_indices, 
                   strides=strides, 
                   dilation_rates=dilation_rates, 
                   frozen_stages=frozen_stages, 
                   input_shape=input_shape, 
                   input_tensor=input_tensor, 
                   dropblock=dropblock, 
                   last_gamma=last_gamma, 
                   avg_down=False, 
                   num_classes=num_classes,                     
                   drop_rate=drop_rate,
                   **kwargs).build_model()


@BACKBONES.register("SEResNeXt101E_64X4D")
def SEResNeXt101E_64X4D(normalization=dict(normalization ='batch_norm', momentum =0.9, epsilon =1e-05, axis = -1, trainable =True), 
                        activation=dict(activation ='relu'), 
                        output_indices=(-1, ), 
                        strides=(2, 2, 2, 2, 2), 
                        dilation_rates=(1, 1, 1, 1, 1), 
                        frozen_stages=(-1, ), 
                        input_shape=None, 
                        input_tensor=None, 
                        dropblock=None, 
                        last_gamma=False, 
                        num_classes=1000, 
                        drop_rate=0.5, 
                        **kwargs):
    return ResNeXt("se_resnext101e_64x4d", 
                   deep_stem=True, 
                   block_fn=bottleneckx, 
                   num_blocks=[3, 4, 23, 3], 
                   stem_filters=32, 
                   cardinality=64,
                   use_se=True,
                   bottleneck_width=4,
                   normalization=normalization, 
                   activation=activation, 
                   output_indices=output_indices, 
                   strides=strides, 
                   dilation_rates=dilation_rates, 
                   frozen_stages=frozen_stages, 
                   input_shape=input_shape, 
                   input_tensor=input_tensor, 
                   dropblock=dropblock, 
                   last_gamma=last_gamma, 
                   avg_down=True, 
                   num_classes=num_classes, 
                   drop_rate=drop_rate,
                   **kwargs).build_model()


def _get_weight_name_map(blocks, use_se):
    name_map = {
        "stem/conv1/conv2d/kernel:0": "resnext0_conv0_weight",
        "stem/conv1/batch_norm/gamma:0": "resnext0_batchnorm0_gamma",
        "stem/conv1/batch_norm/beta:0": "resnext0_batchnorm0_beta",
        "stem/conv1/batch_norm/moving_mean:0": "resnext0_batchnorm0_running_mean",
        "stem/conv1/batch_norm/moving_variance:0": "resnext0_batchnorm0_running_var"
    }

    for i, b in enumerate(blocks, 1):
        n = 0
        for j in range(b):
            k = 3 + 2 if use_se else 3
            for m in range(1, k+1):
                name_map["layer%d/%d/conv%d/conv2d/kernel:0" % (i, j, m)] = "resnext0_stage%d_conv%d_weight" % (i, n)
                name_map["layer%d/%d/conv%d/batch_norm/gamma:0" % (i, j, m)] = "resnext0_stage%d_batchnorm%d_gamma" % (i, n)
                name_map["layer%d/%d/conv%d/batch_norm/beta:0" % (i, j, m)] = "resnext0_stage%d_batchnorm%d_beta" % (i, n)
                name_map["layer%d/%d/conv%d/batch_norm/moving_mean:0" % (i, j, m)] = "resnext0_stage%d_batchnorm%d_running_mean" % (i, n)
                name_map["layer%d/%d/conv%d/batch_norm/moving_variance:0" % (i, j, m)] = "resnext0_stage%d_batchnorm%d_running_var" % (i, n)
                n += 1
            if j == 0:
                name_map["layer%d/%d/shortcut/conv2d/kernel:0" % (i, j)] = "resnext0_stage%d_conv%d_weight" % (i, n)
                name_map["layer%d/%d/shortcut/batch_norm/gamma:0" % (i, j)] = "resnext0_stage%d_batchnorm%d_gamma" % (i, n)
                name_map["layer%d/%d/shortcut/batch_norm/beta:0" % (i, j)] = "resnext0_stage%d_batchnorm%d_beta" % (i, n)
                name_map["layer%d/%d/shortcut/batch_norm/moving_mean:0" % (i, j)] = "resnext0_stage%d_batchnorm%d_running_mean" % (i, n)
                name_map["layer%d/%d/shortcut/batch_norm/moving_variance:0" % (i, j)] = "resnext0_stage%d_batchnorm%d_running_var" % (i, n)
                n += 1
    name_map["logits/kernel:0"] = "resnext0_dense0_weight"
    name_map["logits/bias:0"] = "resnext0_dense0_bias"
    
    return name_map


def _mxnet2h5(model, blocks, name):
    from gluoncv.model_zoo import get_model
    from mxnet import nd, image
    from gluoncv.data.transforms.presets.imagenet import transform_eval

    net = get_model(name, pretrained=True)

    m_weights = net.collect_params()
    # for k, v in m_weights.items():
    #     print(k, v.shape)
    
    img = image.imread("/home/bail/Documents/pandas.jpg")
    img = transform_eval(img)
    net = get_model(name, pretrained=True)
    pred = net(img)
    topK = 5
    ind = nd.topk(pred, k=topK)[0].astype('int')
    print('The input picture is classified to be')
    for i in range(topK):
        print('\t[%s], with probability %.3f.'%
              (ind[i].asscalar(), nd.softmax(pred)[0][ind[i]].asscalar()))

    name_map = _get_weight_name_map(blocks, False)
    # for k, v in name_map.items():
    #     print(k, v)
    for w in model.weights:            
        mw = m_weights[name_map[w.name]].data().asnumpy()
        # print(w.name, w.shape.as_list())
        if len(mw.shape) == 4:
            mw = mw.transpose((2, 3, 1, 0))
        
        if len(mw.shape) == 2:
            mw = mw.transpose((1, 0))
        w.assign(mw)

    del net


if __name__ == '__main__':
    name = "resnext50_32x4d"
    blocks = [3, 4, 6, 3]
    model = ResNeXt50_32X4D(input_shape=(224, 224, 3), output_indices=(-1, ))
    _mxnet2h5(model, blocks, name)
    # model.load_weights("/home/bail/Workspace/pretrained_weights/%s/%s.ckpt" % (name, name))

    with tf.io.gfile.GFile("/home/bail/Documents/pandas.jpg", "rb") as gf:
        images = tf.image.decode_jpeg(gf.read())

    images = tf.cast(images, tf.float32)
    images = tf.image.resize(images, (224, 224))[None]
    logits = model(images, training=False)
    probs = tf.nn.softmax(logits)
    print(tf.nn.top_k(tf.squeeze(probs), k=5))

    model.save_weights("/home/bail/Data/data2/pretrained_weights/%s.h5" % name)
