import os
import tensorflow as tf
from .backbone import Backbone
from ..builder import BACKBONES
from ..common import ConvNormActBlock
from core.layers import build_activation


def bottleneck_v1b(inputs,
                   filters,
                   strides=1,
                   dilation_rate=1,
                   normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                   activation=dict(activation="relu"),
                   trainable=True,
                   dropblock=None,
                   last_gamma_init_zero=False,
                   use_conv_shortcut=False,
                   data_format="channels_last",
                   expansion=4,
                   name="bottleneck_v1b"):
    """A residual block.

        Args:
            filters: integer, filters of the bottleneck layer.
            strides: default 1, stride of the first layer.
            dilation_rate: default 1, dilation rate in 3x3 convolution.
            activation: the activation layer name.
            trainable: does this block is trainable.
            normalization: the normalization, e.g. "batch_norm", "group_norm" etc.
            dropblock: the arguments in DropBlock2D
            use_conv_shortcut: default True, use convolution shortcut if True,
                otherwise identity shortcut.
        Returns:
            Output tensor for the residual block.
    """
    x = ConvNormActBlock(filters=filters, 
                         kernel_size=1, 
                         trainable=trainable, 
                         normalization=normalization,
                         activation=activation,
                         data_format=data_format,
                         dropblock=dropblock,
                         name=name + "/conv1")(inputs)
    x = ConvNormActBlock(filters=filters,
                         kernel_size=3,
                         strides=strides,
                         dilation_rate=1 if strides == 2 else dilation_rate,
                         data_format=data_format,
                         trainable=trainable,
                         normalization=normalization,
                         activation=activation,
                         dropblock=dropblock,
                         name=name + "/conv2")(x)
    x = ConvNormActBlock(filters=filters * expansion,
                         kernel_size=1,
                         data_format=data_format,
                         trainable=trainable,
                         normalization=normalization,
                         gamma_zeros=last_gamma_init_zero,
                         activation=None,
                         name=name + "/conv3")(x)
    
    shortcut = inputs
    if use_conv_shortcut:
        if dilation_rate == 1:
            shortcut = tf.keras.layers.AvgPool2D(pool_size=strides, 
                                                 strides=strides, 
                                                 padding="same", 
                                                 data_format=data_format,
                                                 name=name + "/avgpool")(shortcut)
        else:
            shortcut = tf.keras.layers.AvgPool2D(pool_size=1, 
                                                 strides=1, 
                                                 padding="same", 
                                                 data_format=data_format,
                                                 name=name + "/avgpool")(shortcut)
        shortcut = ConvNormActBlock(filters=filters * expansion,
                                    kernel_size=1, 
                                    data_format=data_format,
                                    trainable=trainable,
                                    normalization=normalization,
                                    dropblock=dropblock,
                                    activation=None,
                                    name=name + "/downsample")(shortcut)
    
    x = tf.keras.layers.Add(name=name + "/add")([x, shortcut])
    x = build_activation(name=name + "/" + activation["activation"], **activation)(x)
    
    return x


class ResNetV1B(Backbone):
    def __init__(self, 
                 name, 
                 deep_stem=False,
                 num_blocks=(3, 4, 6, 3),
                 stem_filters=32,
                 normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                 activation=dict(activation="relu"),
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
        super(ResNetV1B, self).__init__(name,
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
        self.num_blocks = num_blocks
        self.stem_filters = stem_filters
        self.last_gamma = last_gamma

    def build_model(self):
        def _norm(inp):
            mean = tf.constant([0.485, 0.456, 0.406], inp.dtype, [1, 1, 1, 3]) * 255.
            std = 1. / (tf.constant([0.229, 0.224, 0.225], inp.dtype, [1, 1, 1, 3]) * 255.)
            return (inp - mean) * std  

        x = tf.keras.layers.Lambda(_norm, name="norm_input")(self.img_input)

        x = ConvNormActBlock(filters=self.stem_filters,
                             kernel_size=3,
                             strides=self.strides[0],
                             data_format=self.data_format,
                             normalization=self.normalization,
                             activation=self.activation,
                             trainable=1 not in self.frozen_stages,
                             dropblock=self.dropblock,
                             dilation_rate=self.dilation_rates[0],
                             name="conv1/0")(x)
        x = ConvNormActBlock(filters=self.stem_filters,
                             kernel_size=3,
                             data_format=self.data_format,
                             normalization=self.normalization,
                             trainable=1 not in self.frozen_stages,
                             activation=self.activation,
                             dilation_rate=self.dilation_rates[0],
                             dropblock=self.dropblock,
                             name="conv1/1")(x)
        x = ConvNormActBlock(filters=self.stem_filters * 2,
                             kernel_size=3,
                             data_format=self.data_format,
                             normalization=self.normalization,
                             dilation_rate=self.dilation_rates[0],
                             activation=self.activation,
                             trainable=1 not in self.frozen_stages,
                             dropblock=self.dropblock,
                             name="conv1/2")(x)
        outputs = [x]
        x = tf.keras.layers.ZeroPadding2D(((1, 1), (1, 1)), data_format=self.data_format)(x)
        x = tf.keras.layers.MaxPool2D(pool_size=(3, 3), 
                                      strides=self.strides[1], 
                                      padding="valid", 
                                      data_format=self.data_format,
                                      name="max_pool0")(x)
        x = self._make_layers(x, 64, self.num_blocks[0], 1, self.dilation_rates[1], 2 not in self.frozen_stages, "layers1")
        outputs.append(x)
        x = self._make_layers(x, 128, self.num_blocks[1], self.strides[2], self.dilation_rates[2], 3 not in self.frozen_stages, "layers2")
        outputs.append(x)
        x = self._make_layers(x, 256, self.num_blocks[2], self.strides[3], self.dilation_rates[3], 4 not in self.frozen_stages, "layers3")
        outputs.append(x)
        x = self._make_layers(x, 512, self.num_blocks[3], self.strides[4], self.dilation_rates[4], 5 not in self.frozen_stages, "layers4")
        outputs.append(x)
        
        if -1 not in self.output_indices:
            outputs = [outputs[i-1] for i in self.output_indices]
        else:
            x = tf.keras.layers.GlobalAvgPool2D(data_format=self.data_format)(x)
            if self.drop_rate and self.drop_rate > 0.:
                x = tf.keras.layers.Dropout(rate=self.drop_rate)(x)
            
            outputs = tf.keras.layers.Dense(units=self.num_classes, name="logits")(x)

        return tf.keras.Model(inputs=self.img_input, outputs=outputs, name=self.name)

    def _make_layers(self, x, filters, num_blocks, strides=1, dilation_rate=1, trainable=True, name="layer"):
        x = bottleneck_v1b(inputs=x,
                           filters=filters,
                           strides=strides,
                           dilation_rate=dilation_rate,
                           normalization=self.normalization,
                           activation=self.activation,
                           trainable=trainable,
                           dropblock=self.dropblock,
                           last_gamma_init_zero=self.last_gamma,
                           data_format=self.data_format,
                           use_conv_shortcut=True,
                           name=name + "/0")
        for i in range(1, num_blocks):
            x = bottleneck_v1b(inputs=x,
                               filters=filters,
                               strides=1,
                               dilation_rate=dilation_rate,
                               normalization=self.normalization,
                               activation=self.activation,
                               trainable=trainable,
                               dropblock=self.dropblock,
                               last_gamma_init_zero=self.last_gamma,
                               data_format=self.data_format,
                               use_conv_shortcut=False,
                               name=name + "/%d" % i)

        return x


@BACKBONES.register("ResNet50V1D")
def ResNet50V1D(normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                activation=dict(activation="relu"),
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
    return ResNetV1B("resnet50_v1d",
                     normalization=normalization, 
                     activation=activation, 
                     output_indices=output_indices, 
                     strides=strides, 
                     dilation_rates=dilation_rates, 
                     frozen_stages=frozen_stages, 
                     input_shape=input_shape, 
                     input_tensor=input_tensor,
                     dropblock=dropblock, 
                     deep_stem=True,
                     last_gamma=last_gamma,
                     num_blocks=[3, 4, 6, 3],
                     stem_filters=32,
                     num_classes=num_classes,
                     drop_rate=drop_rate,
                     **kwargs).build_model()


@BACKBONES.register("ResNet101V1D")
def ResNet101V1D(normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                 activation=dict(activation="relu"),
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
    return ResNetV1B("resnet101_v1d",
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
                     deep_stem=True,
                     num_blocks=[3, 4, 23, 3],
                     stem_filters=32,
                     num_classes=num_classes,
                     drop_rate=drop_rate,
                     **kwargs).build_model( )
 

@BACKBONES.register("ResNet152V1D")
def ResNet152V1D(normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                 activation=dict(activation="relu"),
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
    return ResNetV1B("resnet152_v1d",
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
                     deep_stem=True,
                     num_blocks=[3, 8, 36, 3],
                     stem_filters=32,
                     num_classes=num_classes,
                     drop_rate=drop_rate,
                     **kwargs).build_model()


@BACKBONES.register("ResNet50V1E")
def ResNet50V1E(normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                activation=dict(activation="relu"),
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
    return ResNetV1B("resnet50_v1e",
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
                     deep_stem=True,
                     num_blocks=[3, 4, 6, 3],
                     stem_filters=64,
                     num_classes=num_classes,
                     drop_rate=drop_rate,
                     **kwargs).build_model()


@BACKBONES.register("ResNet101V1E")
def ResNet101V1E(normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                 activation=dict(activation="relu"),
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
    return ResNetV1B("resnet101_v1e",
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
                     deep_stem=True,
                     num_blocks=[3, 4, 23, 3],
                     stem_filters=64,
                     num_classes=num_classes,
                     drop_rate=drop_rate,
                     **kwargs).build_model()


@BACKBONES.register("ResNet152V1E")
def ResNet152V1E(normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                 activation=dict(activation="relu"),
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
    return ResNetV1B("resnet152_v1e",
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
                     deep_stem=True,
                     num_blocks=[3, 8, 36, 3],
                     stem_filters=64,
                     num_classes=num_classes,
                     drop_rate=drop_rate,
                     **kwargs).build_model()


def _get_weight_name_map(blocks):
    name_map = {
        "conv1/0/conv2d/kernel:0": "conv0_weight",
        "conv1/0/batch_norm/gamma:0": "batchnorm0_gamma",
        "conv1/0/batch_norm/beta:0": "batchnorm0_beta",
        "conv1/0/batch_norm/moving_mean:0": "batchnorm0_running_mean",
        "conv1/0/batch_norm/moving_variance:0": "batchnorm0_running_var",
        "conv1/1/conv2d/kernel:0": "conv1_weight",
        "conv1/1/batch_norm/gamma:0": "batchnorm1_gamma",
        "conv1/1/batch_norm/beta:0": "batchnorm1_beta",
        "conv1/1/batch_norm/moving_mean:0": "batchnorm1_running_mean",
        "conv1/1/batch_norm/moving_variance:0": "batchnorm1_running_var",
        "conv1/2/conv2d/kernel:0": "conv2_weight",
        "conv1/2/batch_norm/gamma:0": "batchnorm2_gamma",
        "conv1/2/batch_norm/beta:0": "batchnorm2_beta",
        "conv1/2/batch_norm/moving_mean:0": "batchnorm2_running_mean",
        "conv1/2/batch_norm/moving_variance:0": "batchnorm2_running_var",
    }

    for i in range(1, 5):
        n = 0
        for j in range(blocks[i - 1]):
            for k in range(1, 4):
                n1 = "layers%d/%d/conv%d" % (i, j, k)
                n2 = "layers%d" % i
                m = {
                    "%s/conv2d/kernel:0" % n1: "%s_conv%d_weight" % (n2, n), 
                    "%s/batch_norm/gamma:0" % n1: "%s_batchnorm%d_gamma" % (n2, n),
                    "%s/batch_norm/beta:0" % n1: "%s_batchnorm%d_beta" % (n2, n),
                    "%s/batch_norm/moving_mean:0" % n1: "%s_batchnorm%d_running_mean" % (n2, n),
                    "%s/batch_norm/moving_variance:0" % n1: "%s_batchnorm%d_running_var" % (n2, n),
                    "layers%d/0/downsample/conv2d/kernel:0" % i: "down%d_conv0_weight" % i,
                    "layers%d/0/downsample/batch_norm/gamma:0" % i: "down%d_batchnorm0_gamma" % i,
                    "layers%d/0/downsample/batch_norm/beta:0" % i: "down%d_batchnorm0_beta" % i,
                    "layers%d/0/downsample/batch_norm/moving_mean:0" % i: "down%d_batchnorm0_running_mean" % i,
                    "layers%d/0/downsample/batch_norm/moving_variance:0" % i: "down%d_batchnorm0_running_var" % i
                }
                name_map.update(m)
                n += 1
    
    name_map["logits/kernel:0"] = "dense0_weight"
    name_map["logits/bias:0"] = "dense0_bias"

    return name_map


def _mxnet2h5(model, name, blocks=(3, 4, 6, 3)):
    from mxnet import nd, image
    from gluoncv.model_zoo import get_model
    from gluoncv.data.transforms.presets.imagenet import transform_eval

    m_name = name
    m_name = m_name.replace("18", "")
    m_name = m_name.replace("35", "")
    m_name = m_name.replace("50", "")
    m_name = m_name.replace("101", "")
    m_name = m_name.replace("152", "")
    m_name = m_name.replace("_", "")

    img = image.imread("/Users/bailang/Documents/pandas.jpg")
    img = transform_eval(img)
    net = get_model(name, pretrained=True)
    pred = net(img)
    topK = 5
    ind = nd.topk(pred, k=topK)[0].astype('int')
    print('The input picture is classified to be')
    for i in range(topK):
        print('\t[%s], with probability %.3f.'%
              (ind[i].asscalar(), nd.softmax(pred)[0][ind[i]].asscalar()))
    mx_weights = net.collect_params()
    # for k, v in mx_weights.items():
    #     print(k)

    name_map = _get_weight_name_map(blocks)
    # for k, v in name_map.items():
    #     print(k, v)
    for weight in model.weights:
        w_name = weight.name 
        mx_name = m_name + "_" + name_map[w_name]
        mx_w = mx_weights[mx_name].data().asnumpy()
        if len(mx_w.shape) == 4:
            mx_w = mx_w.transpose((2, 3, 1, 0))
        if len(mx_w.shape) == 2:
            mx_w = mx_w.transpose((1, 0))

        weight.assign(mx_w)


if __name__ == "__main__":
    name = "resnet50_v1e"
    blocks = (3, 4, 6, 3)
    model = ResNet50V1E(input_shape=(224, 224, 3), output_indices=(-1, ))
    # model.summary()
    _mxnet2h5(model, name, blocks)
    
    with tf.io.gfile.GFile("/Users/bailang/Documents/pandas.jpg", "rb") as gf:
        images = tf.image.decode_jpeg(gf.read())

    images = tf.image.resize(images, (224, 224))
    images = tf.expand_dims(images, axis=0)
    lbl = model(images, training=False)
    top5prob, top5class = tf.nn.top_k(tf.squeeze(tf.nn.softmax(lbl, -1), axis=0), k=5)
    print("prob:", top5prob.numpy())
    print("class:", top5class.numpy())
    
    model.save_weights("/Users/bailang/Downloads/pretrained_weights/%s.h5" % name)
    model.save_weights("/Users/bailang/Downloads/pretrained_weights/%s/model.ckpt" % name)

