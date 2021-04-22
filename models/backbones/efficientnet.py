import os
import math
import tensorflow as tf
from collections import namedtuple
from models.backbones.backbone import Backbone
from core.layers import build_normalization
from models.builder import BACKBONES


PARAMS = {
        # (width_coefficient, depth_coefficient, resolution, dropout_rate)
        "efficientnet-b0": (1.0, 1.0, 224, 0.2),
        "efficientnet-b1": (1.0, 1.1, 240, 0.2),
        "efficientnet-b2": (1.1, 1.2, 260, 0.3),
        "efficientnet-b3": (1.2, 1.4, 300, 0.3),
        "efficientnet-b4": (1.4, 1.8, 380, 0.4),
        "efficientnet-b5": (1.6, 2.2, 456, 0.4),
        "efficientnet-b6": (1.8, 2.6, 528, 0.5),
        "efficientnet-b7": (2.0, 3.1, 600, 0.5),
    }


GlobalParams = namedtuple("GlobalParams", [
    "batch_norm_momentum", "batch_norm_epsilon", "width_coefficient",
    "depth_coefficient", "depth_divisor", "min_depth", "drop_connect_rate",
    "data_format", "dropout_rate", "num_classes"
])
GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)

BlockArgs = namedtuple("BlockArgs", [
    "repeats", "in_filters", "out_filters", "kernel_size",
    "strides", "expand_ratio", "se_ratio", "id_skip", "super_pixel", "trainable"
])
BlockArgs.__new__.__defaults__ = (None, ) * len(BlockArgs._fields)


def round_filters(filters, global_params):
    # orig_filters =filters

    multiplier = global_params.width_coefficient
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth

    if multiplier is None:
        return filters

    min_depth = min_depth or divisor
    filters *= multiplier
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)

    if new_filters < 0.9 * filters:
        new_filters += divisor

    return int(new_filters)


def round_repeats(repeats, global_params):
    multiplier = global_params.depth_coefficient

    if multiplier is None:
        return repeats

    return int(math.ceil(repeats * multiplier))


class DropConnect(tf.keras.layers.Layer):
    def __init__(self, drop_rate=None, **kwargs):
        super(DropConnect, self).__init__(**kwargs)
        self.drop_rate = drop_rate if drop_rate is not None else 0.

    def _drop(self, inputs, drop_rate):
        random_tensor = tf.convert_to_tensor(drop_rate, dtype=inputs.dtype)
        batch_size = tf.shape(inputs)[0]
        random_tensor += tf.random.uniform([batch_size, 1, 1, 1], dtype=inputs.dtype)
        binary_tensor = tf.math.floor(random_tensor)

        return tf.divide(inputs, random_tensor) * binary_tensor

    def call(self, inputs, training=None):
        if training or self.drop_rate > 0.:
            return self._drop(inputs, self.drop_rate)
        return inputs

    def compute_output_shape(self, input_shape):
        return input_shape


def conv2d_kernel_initializer(shape, dtype=tf.float32):
    kernel_height, kernel_width, _, out_filters = shape
    fan_out = int(kernel_height * kernel_width * out_filters)

    return tf.random.normal(shape, 0.0, math.sqrt(2. / fan_out), dtype=dtype)


def dense_kernel_initializer(shape, dtype=tf.float32):
    init_range = 1.0 / math.sqrt(shape[1])

    return tf.random.uniform(shape, -init_range, init_range, dtype)


def mbconv_block(inputs,
                 global_params,
                 block_args,
                 normalization,
                 drop_connect_rate=None,
                 trainable=True,
                 name=""):
    expand_ratio = block_args.expand_ratio
    data_format = global_params.data_format

    _momentum = global_params.batch_norm_momentum
    _epsilon = global_params.batch_norm_epsilon
    _axis = normalization["axis"]
    _mean_axis = [1, 2] if _axis == -1 or _axis == 3 else [2, 3]
    filters = block_args.in_filters * expand_ratio
    expand_ratio = expand_ratio
    if expand_ratio != 1:
        x = tf.keras.layers.Conv2D(filters=filters,
                                   kernel_size=(1, 1),
                                   strides=(1, 1),
                                   padding="same",
                                   data_format=data_format,
                                   use_bias=False,
                                   name=name+"/conv2d",
                                   trainable=trainable,
                                   kernel_initializer=conv2d_kernel_initializer)(inputs)
        x = build_normalization(**normalization,
                                name=name+"/batch_normalization")(x)
        x = tf.keras.layers.Activation("swish", name=name+"/swish")(x)
    else:
        x = inputs

    # Depthwise Convolution
    # if block_args.strides == 2:
    #     x = tf.keras.layers.ZeroPadding2D(
    #         padding=imagenet_utils.correct_pad(x, block_args.kernel_size),
    #         name=name + '/dwconv_pad')(x)
    #     conv_pad = 'valid'
    # else:
    #     conv_pad = 'same'
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=block_args.kernel_size,
                                        strides=block_args.strides,
                                        padding="same",
                                        data_format=data_format,
                                        use_bias=False,
                                        trainable=trainable,
                                        depthwise_initializer=conv2d_kernel_initializer,
                                        name=name+"/depthwise_conv2d")(x)
    x = build_normalization(**normalization,
                            name=name+"/batch_normalization"
                            if expand_ratio == 1 else name+"/batch_normalization_1")(x)
    x = tf.keras.layers.Activation("swish", name=name+"/swish" if expand_ratio == 1 else name+"/swish_1")(x)

    has_se = block_args.se_ratio is not None and 0 < block_args.se_ratio < 1
    if has_se:
        squeeze_filters = max(1, int(block_args.in_filters * block_args.se_ratio))
        se = tf.keras.layers.Lambda(lambda inp: tf.reduce_mean(inp, axis=_mean_axis, keepdims=True),
                                    name=name+"/se/global_pooling")(x)
        se = tf.keras.layers.Conv2D(filters=squeeze_filters,
                                    kernel_size=(1, 1),
                                    strides=(1, 1),
                                    padding="same",
                                    data_format=data_format,
                                    use_bias=True,
                                    kernel_initializer=conv2d_kernel_initializer,
                                    trainable=trainable,
                                    name=name + "/se/conv2d")(se)
        se = tf.keras.layers.Activation("swish", name=name + "/se/swish_1")(se)
        se = tf.keras.layers.Conv2D(filters=filters,
                                    kernel_size=(1, 1),
                                    strides=(1, 1),
                                    padding="same",
                                    data_format=data_format,
                                    use_bias=True,
                                    trainable=trainable,
                                    kernel_initializer=conv2d_kernel_initializer,
                                    name=name + "/se/conv2d_1")(se)
        se = tf.keras.layers.Activation("sigmoid", name=name+"/se/sigmoid")(se)
        x = tf.keras.layers.Multiply(name=name + "/se/multiply")([se, x])

    x = tf.keras.layers.Conv2D(block_args.out_filters,
                               kernel_size=(1, 1),
                               strides=(1, 1),
                               padding="same",
                               data_format=data_format,
                               use_bias=False,
                               trainable=trainable,
                               kernel_initializer=conv2d_kernel_initializer,
                               name=name+"/conv2d" if expand_ratio == 1 else name+"/conv2d_1")(x)
    x = build_normalization(**normalization,
                            name=name+"/batch_normalization_2"
                            if expand_ratio > 1 else name+"/batch_normalization_1")(x)
    if block_args.id_skip:
        if all(s == 1 for s in block_args.strides) and block_args.in_filters == block_args.out_filters:
            # x = DropConnect(drop_connect_rate, name=name + "/drop_connect")(x)
            # x = tf.keras.layers.Dropout(drop_connect_rate, noise_shape=(None, 1, 1, 1), name=name + '/drop')(x)
            x = tf.keras.layers.Add(name=name + "/add")([x, inputs])

    return x


class EfficientNet(Backbone):
    def _get_global_params(self, name, data_format):
        return GlobalParams(
            batch_norm_momentum=0.9,
            batch_norm_epsilon=1e-3,
            width_coefficient=PARAMS[name][0],
            depth_coefficient=PARAMS[name][1],
            depth_divisor=8,
            min_depth=None,
            drop_connect_rate=0.2,
            data_format=data_format,
            dropout_rate=PARAMS[name][-1],
            num_classes=1000
        )

    def _get_block_args(self):
        return [
            BlockArgs(1, 32, 16, (3, 3), (1, 1), 1, 0.25, True),
            BlockArgs(2, 16, 24, (3, 3), (2, 2), 6, 0.25, True),
            BlockArgs(2, 24, 40, (5, 5), (2, 2), 6, 0.25, True),
            BlockArgs(3, 40, 80, (3, 3), (2, 2), 6, 0.25, True),
            BlockArgs(3, 80, 112, (5, 5), (1, 1), 6, 0.25, True),
            BlockArgs(4, 112, 192, (5, 5), (2, 2), 6, 0.25, True),
            BlockArgs(1, 192, 320, (3, 3), (1, 1), 6, 0.25, True)
        ]

    def __init__(self,
                 name,
                 convolution='conv2d', 
                 normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-3, axis=-1, trainable=True), 
                 activation=dict(activation="swish"), 
                 output_indices=(3, 4), 
                 strides=(2, 2, 2, 2, 2), 
                 dilation_rates=(1,1,1,1,1), 
                 frozen_stages=( -1), 
                 dropblock=None, 
                 input_shape=None,
                 input_tensor=None,
                 **kwargs):
        data_format = tf.keras.backend.image_data_format()
        self.bn_axis = 3 if data_format == 'channels_last' else 1
        default_size = PARAMS[name][2]
        dropout_connect_rate = PARAMS[name][3]
        default_shape = [default_size, default_size, 3] if self.bn_axis == 3 else [3, default_size, default_size]
        input_shape = input_shape or default_shape

        super(EfficientNet, self).__init__(name=name,
                                           convolution=convolution,
                                           normalization=normalization, 
                                           activation=activation, 
                                           output_indices=output_indices, 
                                           strides=strides, 
                                           dilation_rates=dilation_rates, 
                                           frozen_stages=frozen_stages, 
                                           dropblock=dropblock, 
                                           input_shape=input_shape,
                                           input_tensor=input_tensor)
        self.backbone_name = name
        self.data_format = data_format

        self.global_params = self._get_global_params(name, self.data_format)
        self.block_args = self._get_block_args()

        self._drop_connect_rate = dropout_connect_rate

        self.num_blocks = 0
        for args in self.block_args:
            self.num_blocks += round_repeats(args.repeats, global_params=self.global_params)
    
    @property
    def blocks(self):
        blocks = []
        for i, args in enumerate(self.block_args):
            assert args.repeats >= 1
            # assert args.super_pixel in [0, 1, 2]
            in_filters = round_filters(args.in_filters, self.global_params)
            out_filters = round_filters(args.out_filters, self.global_params)
            
            args = args._replace(in_filters=in_filters,
                                 out_filters=out_filters,
                                 repeats=round_repeats(args.repeats, self.global_params),
                                 trainable=i + 2 not in self.frozen_stages)
            blocks.append(args)
            if args.repeats > 1:
                args = args._replace(in_filters=out_filters, strides=(1, 1))
            for i in range(args.repeats - 1):
                blocks.append(args)
        
        return blocks
    
    def build_model(self):
        def _norm(inp):
            mean = tf.constant([0.485, 0.456, 0.406], inp.dtype, [1, 1, 1, 3]) * 255.
            std = 1. / (tf.constant([0.229, 0.224, 0.225], inp.dtype, [1, 1, 1, 3]) * 255.)

            return (inp - mean) * std

        x = tf.keras.layers.Lambda(_norm, name="norm_input")(self.img_input)
    
        x = tf.keras.layers.Conv2D(round_filters(32, self.global_params),
                                   kernel_size=(3, 3),
                                   strides=(2, 2),
                                   padding="same",
                                   data_format=self.data_format,
                                   use_bias=False,
                                   kernel_initializer=conv2d_kernel_initializer,
                                   trainable=1 not in self.frozen_stages,
                                   name=self.name + "/stem/conv2d")(x)
        x = build_normalization(**self.normalization, name=self.name + "/stem/batch_normalization")(x)
        x = tf.keras.layers.Activation("swish", name=self.name + "/stem/swish")(x)

        block_outputs = []
        for idx, b_args in enumerate(self.blocks):
            drop_rate = self._drop_connect_rate
            is_reduction = False
                
            if b_args.super_pixel == 1 and idx == 0:
                block_outputs.append(x)
            elif (idx == self.num_blocks - 1) or self.blocks[idx+1].strides[0] > 1:
                is_reduction = True
            if drop_rate:
                drop_rate = 1.0 - drop_rate * float(idx) / self.num_blocks
                
            x = mbconv_block(x,
                             global_params=self.global_params,
                             block_args=b_args,
                             normalization=self.normalization,
                             drop_connect_rate=drop_rate,
                             trainable=b_args.trainable,
                             name=self.name + "/blocks_%d" % idx)
            if is_reduction:
                block_outputs.append(x)
        
        if -1 in self.output_indices:
            # Head part.
            x = tf.keras.layers.Conv2D(filters=round_filters(1280, self.global_params),
                                       kernel_size=[1, 1],
                                       strides=[1, 1],
                                       kernel_initializer=conv2d_kernel_initializer,
                                       padding="same",
                                       use_bias=False,
                                       data_format=self.data_format,
                                       name=self.name + "/head/conv2d")(x)
            x = build_normalization(**self.normalization, name=self.name + "/head/batch_normalization")(x)
            x = tf.keras.layers.Activation("swish", name=self.name + "/head/swish")(x)
            x = tf.keras.layers.GlobalAveragePooling2D(data_format=self.data_format, 
                                                       name=self.name + "/head/global_avg_pooling")(x)
            x = tf.keras.layers.Dropout(self.global_params.dropout_rate, name=self.name + "/head/dropout")(x)
            x = tf.keras.layers.Dense(self.global_params.num_classes,
                                      kernel_initializer=dense_kernel_initializer, 
                                      name=self.name + "/head/dense")(x)
            outputs = x
            
        else:
            outputs = [block_outputs[i - 1] for i in self.output_indices]
            
        return tf.keras.Model(inputs=self.img_input, outputs=outputs, name=self.name)
    
    def init_weights(self, pretrained_weight_path=None):
        if pretrained_weight_path is not None:
            pretrained_weights = tf.train.latest_checkpoint(pretrained_weight_path)
            assert pretrained_weights is not None, "Error! Please check path {}".format(pretrained_weight_path)
            # use_exponential_moving_average = False
            # for w in tf.train.list_variables(pretrained_weights):
            #     if "ExponentialMovingAverage" not in w[0]:
            #         # use_exponential_moving_average = True
            #         if "box_net" in w[0]:
            #              print(w[0], w[1])

            for weight in self.model.weights:
                name = weight.name.split(":")[0]
                # print(name, weight.shape)
                # if "box-predict" in name or "class-predict" in name:
                #     continue
                if "batch_normalization" in name:
                    name = name.replace("batch_normalization", "tpu_batch_normalization")
                # if use_exponential_moving_average:
                #     name += "/ExponentialMovingAverage"
                try:
                    pretrained_weight = tf.train.load_variable(pretrained_weights, name)
                    weight.assign(pretrained_weight)
                except Exception as e:
                    print(str(e), "{} not in {}.".format(name, pretrained_weight_path))


@BACKBONES.register("EfficientNetB0")
def EfficientNetB0(input_shape,
                   convolution='conv2d', 
                   normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-3, axis=-1, trainable=True), 
                   activation=dict(activation="swish"), 
                   output_indices=(3, 4), 
                   strides=(2, 2, 2, 2, 2), 
                   dilation_rates=(1, 1, 1, 1, 1), 
                   frozen_stages=(-1, ), 
                   dropblock=None, 
                   input_tensor=None,
                   classifier_activation='softmax'):
    return EfficientNet(name="efficientnet-b0",
                        convolution=convolution, 
                        normalization=normalization,
                        activation=activation, 
                        output_indices=output_indices, 
                        strides=strides, 
                        dilation_rates=dilation_rates, 
                        frozen_stages=frozen_stages, 
                        dropblock=dropblock, 
                        input_shape=input_shape,
                        input_tensor=input_tensor,
                        classifier_activation=classifier_activation).build_model()


@BACKBONES.register("EfficientNetB1")
def EfficientNetB1(input_shape,
                   convolution='conv2d', 
                   normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-3, axis=-1, trainable=True), 
                   activation=dict(activation="swish"), 
                   output_indices=(3, 4), 
                   strides=(2, 2, 2, 2, 2), 
                   dilation_rates=(1, 1, 1, 1, 1), 
                   frozen_stages=( -1), 
                   dropblock=None, 
                   input_tensor=None,
                   classifier_activation="softmax"):
    return EfficientNet(name="efficientnet-b1",
                        convolution=convolution, 
                        normalization=normalization,
                        activation=activation, 
                        output_indices=output_indices, 
                        strides=strides, 
                        dilation_rates=dilation_rates, 
                        frozen_stages=frozen_stages, 
                        dropblock=dropblock, 
                        input_shape=input_shape,
                        input_tensor=input_tensor,
                        classifier_activation=classifier_activation).build_model()


@BACKBONES.register("EfficientNetB2")
def EfficientNetB2(input_shape,
                   convolution='conv2d', 
                   normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-3, axis=-1, trainable=True), 
                   activation=dict(activation="swish"), 
                   output_indices=(3, 4), 
                   strides=(2, 2, 2, 2, 2), 
                   dilation_rates=(1, 1, 1, 1, 1), 
                   frozen_stages=( -1), 
                   dropblock=None, 
                   input_tensor=None,
                   classifier_activation="softmax"):
    return EfficientNet(name="efficientnet-b2",
                        convolution=convolution, 
                        normalization=normalization,
                        activation=activation, 
                        output_indices=output_indices, 
                        strides=strides, 
                        dilation_rates=dilation_rates, 
                        frozen_stages=frozen_stages, 
                        dropblock=dropblock, 
                        input_shape=input_shape,
                        input_tensor=input_tensor,
                        classifier_activation=classifier_activation).build_model()


@BACKBONES.register("EfficientNetB3")
def EfficientNetB3(input_shape,
                   convolution='conv2d', 
                   normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-3, axis=-1, trainable=True), 
                   activation=dict(activation="swish"), 
                   output_indices=(3, 4), 
                   strides=(2, 2, 2, 2, 2), 
                   dilation_rates=(1, 1, 1, 1, 1), 
                   frozen_stages=( -1), 
                   dropblock=None, 
                   input_tensor=None,
                   classifier_activation="softmax"):
    return EfficientNet(name="efficientnet-b3",
                        convolution=convolution, 
                        normalization=normalization,
                        activation=activation, 
                        output_indices=output_indices, 
                        strides=strides, 
                        dilation_rates=dilation_rates, 
                        frozen_stages=frozen_stages, 
                        dropblock=dropblock, 
                        input_shape=input_shape,
                        input_tensor=input_tensor,
                        classifier_activation=classifier_activation).build_model()



@BACKBONES.register("EfficientNetB4")
def EfficientNetB4(input_shape,
                   convolution='conv2d', 
                   normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-3, axis=-1, trainable=True), 
                   activation=dict(activation="swish"), 
                   output_indices=(3, 4), 
                   strides=(2, 2, 2, 2, 2), 
                   dilation_rates=(1, 1, 1, 1, 1), 
                   frozen_stages=( -1), 
                   dropblock=None, 
                   input_tensor=None,
                   classifier_activation="softmax"):
    return EfficientNet(name="efficientnet-b4",
                        convolution=convolution, 
                        normalization=normalization,
                        activation=activation, 
                        output_indices=output_indices, 
                        strides=strides, 
                        dilation_rates=dilation_rates, 
                        frozen_stages=frozen_stages, 
                        dropblock=dropblock, 
                        input_shape=input_shape,
                        input_tensor=input_tensor,
                        classifier_activation=classifier_activation).build_model()


@BACKBONES.register("EfficientNetB5")
def EfficientNetB5(input_shape,
                    convolution='conv2d', 
                    normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-3, axis=-1, trainable=True), 
                    activation=dict(activation="swish"), 
                    output_indices=(3, 4), 
                    strides=(2, 2, 2, 2, 2), 
                    dilation_rates=(1, 1, 1, 1, 1), 
                    frozen_stages=( -1), 
                    dropblock=None, 
                    input_tensor=None,
                    classifier_activation="softmax"):
    return EfficientNet(name="efficientnet-b5",
                        convolution=convolution, 
                        normalization=normalization,
                        activation=activation, 
                        output_indices=output_indices, 
                        strides=strides, 
                        dilation_rates=dilation_rates, 
                        frozen_stages=frozen_stages, 
                        dropblock=dropblock, 
                        input_shape=input_shape,
                        input_tensor=input_tensor,
                        classifier_activation=classifier_activation).build_model()


@BACKBONES.register("EfficientNetB6")
def EfficientNetB6(input_shape,
                   convolution='conv2d', 
                   normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-3, axis=-1, trainable=True), 
                   activation=dict(activation="swish"), 
                   output_indices=(3, 4), 
                   strides=(2, 2, 2, 2, 2), 
                   dilation_rates=(1, 1, 1, 1, 1), 
                   frozen_stages=( -1), 
                   dropblock=None, 
                   input_tensor=None,
                   classifier_activation="softmax"):
    return EfficientNet(name="efficientnet-b6",
                        convolution=convolution, 
                        normalization=normalization,
                        activation=activation, 
                        output_indices=output_indices, 
                        strides=strides, 
                        dilation_rates=dilation_rates, 
                        frozen_stages=frozen_stages, 
                        dropblock=dropblock, 
                        input_shape=input_shape,
                        input_tensor=input_tensor,
                        classifier_activation=classifier_activation).build_model()


@BACKBONES.register("EfficientNetB7")
def EfficientNetB7(input_shape,
                   convolution='conv2d', 
                   normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-3, axis=-1, trainable=True), 
                   activation=dict(activation="swish"), 
                   output_indices=(3, 4), 
                   strides=(2, 2, 2, 2, 2), 
                   dilation_rates=(1, 1, 1, 1, 1), 
                   frozen_stages=( -1), 
                   dropblock=None, 
                   input_tensor=None,
                   classifier_activation="softmax"):
    return EfficientNet(name="efficientnet-b7",
                        convolution=convolution, 
                        normalization=normalization,
                        activation=activation, 
                        output_indices=output_indices, 
                        strides=strides, 
                        dilation_rates=dilation_rates, 
                        frozen_stages=frozen_stages, 
                        dropblock=dropblock, 
                        input_shape=input_shape,
                        input_tensor=input_tensor,
                        classifier_activation=classifier_activation).build_model()


if __name__ == "__main__":
    shape = 300
    checkpoint_dir = "/home/bail/Workspace/pretrained_weights/"
    efficientnet = EfficientNetB0(input_shape=(shape, shape, 3), output_indices=[-1], frozen_stages=(0, 1, 2))

    with tf.io.gfile.GFile("/home/bail/Documents/pandas.jpg", "rb") as gf:
        images = tf.image.decode_jpeg(gf.read())

    images = tf.image.resize(images, (shape, shape))[None]
    model = efficientnet.build_model()
    print(images.shape)
    cls = model(images, training=False)
    
    print(tf.argmax(tf.squeeze(cls)))
    print(tf.reduce_max(cls))
    print(tf.nn.top_k(tf.squeeze(cls), k=5))
    print(tf.add_n(model.losses))