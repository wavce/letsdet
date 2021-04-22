import os
import numpy as np
import tensorflow as tf
from .backbone import Backbone
from ..builder import BACKBONES
from ..common import ConvNormActBlock
from core.layers import build_activation


def basic_block(inputs,
                filters,
                strides=1,
                dilation_rate=1,
                data_format="channels_last",
                normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-3, axis=-1, trainable=True),
                activation=dict(activation="relu"),
                trainable=True,
                dropblock=None,
                use_conv_shortcut=False,
                expansion=1,
                strides_in_1x1=False,  # not use
                name=None):
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
    x = ConvNormActBlock(filters=filters,
                         kernel_size=3,
                         strides=strides,
                         data_format=data_format,
                         dilation_rate=1 if strides > 1 else dilation_rate,
                         trainable=trainable,
                         normalization=normalization,
                         activation=activation,
                         dropblock=dropblock,
                         name=name + "/conv1")(inputs)
    x = ConvNormActBlock(filters=filters,
                         kernel_size=3,
                         strides=1,
                         data_format=data_format,
                         dilation_rate=dilation_rate,
                         trainable=trainable,
                         normalization=normalization,
                         activation=None,
                         dropblock=dropblock,
                         name=name + "/conv2")(x)
    
    shortcut = inputs
    if use_conv_shortcut:
        shortcut = ConvNormActBlock(filters=filters,
                                    kernel_size=1,
                                    strides=strides,
                                    data_format=data_format,
                                    trainable=trainable,
                                    normalization=normalization,
                                    activation=None,
                                    dropblock=dropblock,
                                    name=name + "/shortcut")(shortcut)
    x = tf.keras.layers.Add(name=name + "/add")([x, shortcut])
    x = build_activation(**activation, name=name + "/" + activation["activation"])(x)
   
    return x


def bottleneck(inputs,
               filters,
               strides=1,
               dilation_rate=1,
               data_format="channels_last",
               normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-3, axis=-1, trainable=True),
               activation=dict(activation="relu"),
               trainable=True,
               dropblock=None,
               use_conv_shortcut=True,
               strides_in_1x1=False,
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
    
    strides_1x1, strides_3x3 = (strides, 1) if strides_in_1x1 else (1, strides)
    x = ConvNormActBlock(filters=filters,
                         kernel_size=1,
                         strides=strides_1x1,
                         trainable=trainable,
                         dropblock=dropblock,
                         data_format=data_format,
                         normalization=normalization,
                         activation=activation,
                         name=name + "/conv1")(inputs)
    x = ConvNormActBlock(filters=filters,
                         kernel_size=3,
                         strides=strides_3x3,
                         dilation_rate=dilation_rate if strides == 1 or strides_1x1 else 1,
                         trainable=trainable,
                         data_format=data_format,
                         normalization=normalization,
                         activation=activation,
                         name=name + "/conv2")(x)
    x = ConvNormActBlock(filters=expansion * filters,
                         kernel_size=1,
                         trainable=trainable,
                         data_format=data_format,
                         normalization=normalization,
                         activation=None,
                         name=name + "/conv3")(x)
    
    shortcut = inputs
    if use_conv_shortcut is True:
        shortcut = ConvNormActBlock(filters=expansion * filters,
                                    kernel_size=1,
                                    strides=strides,
                                    data_format=data_format,
                                    trainable=trainable,
                                    dropblock=dropblock,
                                    normalization=normalization,
                                    activation=None,
                                    name=name + "/shortcut")(shortcut)
    x = tf.keras.layers.Add(name=name + "/add")([x, shortcut])
    x = build_activation(**activation, name=name + "/" + activation["activation"])(x)
    
    return x


class ResNet(Backbone):
    def __init__(self,
                 name,
                 blocks,
                 block_fn,
                 normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                 activation=dict(activation="relu"),
                 output_indices=(-1, ),
                 strides=(2, 2, 2, 2, 2),
                 dilation_rates=(1, 1, 1, 1, 1),
                 frozen_stages=(-1,),
                 dropblock=None,
                 num_classes=1000,
                 drop_rate=0.5,
                 input_shape=(224, 224, 3),
                 input_tensor=None,
                 strides_in_1x1=False,
                 expansion=2,
                 **kwargs):
        super(ResNet, self).__init__(name=name,
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
        self.blocks = blocks
        self.block_fn = block_fn
        self.expansion = expansion
        self.strides_in_1x1 = strides_in_1x1
        if strides_in_1x1:
            self._rgb_mean = np.array([[[]]])
            self._rgb_std = np.array([[[[1., 1., 1.]]]])

    def build_model(self):
        strides_in_1x1 = self.strides_in_1x1
        
        def _norm(inp):
            if strides_in_1x1:
                # mean = tf.constant([102.9801, 115.9465, 122.7717], inp.dtype, [1, 1, 1, 3])
                mean = tf.constant([103.5300, 116.2800, 123.6750], inp.dtype, [1, 1, 1, 3])
                std = tf.constant([1, 1, 1], inp.dtype, [1, 1, 1, 3])
            else:
                mean = tf.constant([0.485, 0.456, 0.406], inp.dtype, [1, 1, 1, 3]) * 255.
                std = 1. / (tf.constant([0.229, 0.224, 0.225], inp.dtype, [1, 1, 1, 3]) * 255.)
            x = (inp - mean) * std 
            
            return x

        x = tf.keras.layers.Lambda(function=_norm, name="norm_input")(self.img_input)

        x = ConvNormActBlock(filters=64,
                             kernel_size=(7, 7),
                             strides=self.strides[0],
                             dilation_rate=self.dilation_rates[0],
                             trainable=1 not in self.frozen_stages,
                             kernel_initializer="he_normal",
                             normalization=self.normalization,
                             data_format=self.data_format,
                             name="stem/conv1")(x)
        x = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)), data_format=self.data_format)(x)
        x1 = tf.keras.layers.MaxPool2D((3, 3), self.strides[1], "valid", data_format=self.data_format, name="pool1")(x)
        self.in_filters = 64

        trainable = 2 not in self.frozen_stages
        x2 = self.stack(x1, 64, 1, self.dilation_rates[1], trainable, self.blocks[0], "layer1")
        trainable = 3 not in self.frozen_stages
        x3 = self.stack(x2, 128, self.strides[2], self.dilation_rates[2], trainable, self.blocks[1], "layer2")
        trainable = 4 not in self.frozen_stages
        x4 = self.stack(x3, 256, self.strides[3], self.dilation_rates[3], trainable, self.blocks[2], "layer3")
        trainable = 5 not in self.frozen_stages
        x5 = self.stack(x4, 512, self.strides[4], self.dilation_rates[4], trainable, self.blocks[3], "layer4")

        if -1 in self.output_indices:
            x = tf.keras.layers.GlobalAvgPool2D(data_format=self.data_format, name="avg_pool")(x5)
            x = tf.keras.layers.Dropout(rate=self.drop_rate)(x)
            outputs = tf.keras.layers.Dense(self.num_classes, name="logits")(x)
        else:
            outputs = [o for i, o in enumerate([x1, x2, x3, x4, x5]) if i + 1 in self.output_indices]

        model = tf.keras.Model(inputs=self.img_input, outputs=outputs, name=self.name)

        return model

    def stack(self, x, filters, strides, dilation_rate, trainable, blocks, name=None):
        use_conv_shortcut = False
        if strides != 1 or self.in_filters != filters * self.expansion:
            use_conv_shortcut = True
        x = self.block_fn(inputs=x,
                          filters=filters,
                          strides=strides,
                          dilation_rate=dilation_rate,
                          normalization=self.normalization,
                          activation=self.activation,
                          trainable=trainable,
                          dropblock=self.dropblock,
                          expansion=self.expansion,
                          use_conv_shortcut=use_conv_shortcut,
                          strides_in_1x1=self.strides_in_1x1,
                          data_format=self.data_format,
                          name=name + "/0")
        for i in range(1, blocks):
            x = self.block_fn(inputs=x,
                              filters=filters,
                              strides=1,
                              dilation_rate=dilation_rate,
                              normalization=self.normalization,
                              activation=self.activation,
                              trainable=trainable,
                              dropblock=self.dropblock,
                              expansion=self.expansion,
                              use_conv_shortcut=False,
                              data_format=self.data_format,
                              name=name + "/%d" % i)
        return x


@BACKBONES.register("ResNet18")
def ResNet18(normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
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
    return ResNet(name="resnet18",
                  blocks=[2, 2, 2, 2],
                  block_fn=basic_block,
                  normalization=normalization,
                  activation=activation,
                  strides=strides,
                  dilation_rates=dilation_rates,
                  frozen_stages=frozen_stages,
                  dropblock=dropblock,
                  num_classes=num_classes,
                  drop_rate=drop_rate,
                  expansion=1,
                  output_indices=output_indices,
                  input_shape=input_shape,
                  input_tensor=input_tensor,
                  **kwargs).build_model()   


@BACKBONES.register("ResNet34")
def ResNet34(normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
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
    return ResNet(name="resnet34",
                  blocks=[3, 4, 6, 3],
                  block_fn=basic_block,
                  normalization=normalization,
                  activation=activation,
                  strides=strides,
                  expansion=1,
                  output_indices=output_indices,
                  dilation_rates=dilation_rates,
                  frozen_stages=frozen_stages,
                  dropblock=dropblock,
                  num_classes=num_classes,
                  drop_rate=drop_rate,
                  input_shape=input_shape,
                  input_tensor=input_tensor,
                  **kwargs).build_model()   


@BACKBONES.register("ResNet50")
def ResNet50(normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
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

    return ResNet(name="resnet50",
                  blocks=[3, 4, 6, 3],
                  block_fn=bottleneck,
                  normalization=normalization,
                  activation=activation,
                  output_indices=output_indices,
                  strides=strides,
                  dilation_rates=dilation_rates,
                  frozen_stages=frozen_stages,
                  dropblock=dropblock,
                  num_classes=num_classes,
                  input_shape=input_shape,
                  expansion=4,
                  input_tensor=input_tensor,
                  drop_rate=drop_rate,
                  **kwargs).build_model()


@BACKBONES.register("ResNet101")
def ResNet101(normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
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
    return ResNet(name="resnet101",
                  blocks=[3, 4, 23, 3],
                  block_fn=bottleneck,
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
                  expansion=4,
                  drop_rate=drop_rate,
                  **kwargs).build_model()


@BACKBONES.register("ResNet152")
def ResNet152(normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
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
    return ResNet(name="resnet152",
                  blocks=[3, 8, 36, 3],
                  block_fn=bottleneck,
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
                  expansion=4,
                  **kwargs).build_model()


@BACKBONES.register("CaffeResNet50")
def CaffeResNet50(normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
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

    return ResNet(name="resnet50",
                  blocks=[3, 4, 6, 3],
                  block_fn=bottleneck,
                  normalization=normalization,
                  activation=activation,
                  output_indices=output_indices,
                  strides=strides,
                  dilation_rates=dilation_rates,
                  frozen_stages=frozen_stages,
                  dropblock=dropblock,
                  num_classes=num_classes,
                  input_shape=input_shape,
                  expansion=4,
                  strides_in_1x1=True,
                  input_tensor=input_tensor,
                  drop_rate=drop_rate,
                  **kwargs).build_model()


@BACKBONES.register("CaffeResNet101")
def CaffeResNet101(normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
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
    return ResNet(name="resnet101",
                  blocks=[3, 4, 23, 3],
                  block_fn=bottleneck,
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
                  strides_in_1x1=True,
                  expansion=4,
                  drop_rate=drop_rate,
                  **kwargs).build_model()


@BACKBONES.register("CaffeResNet152")
def CaffeResNet152(normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
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
    return ResNet(name="resnet152",
                  blocks=[3, 8, 36, 3],
                  block_fn=bottleneck,
                  normalization=normalization,
                  activation=activation,
                  output_indices=output_indices,
                  strides=strides,
                  dilation_rates=dilation_rates,
                  frozen_stages=frozen_stages,
                  dropblock=dropblock,
                  num_classes=num_classes,
                  input_shape=input_shape,
                  strides_in_1x1=True,
                  input_tensor=input_tensor,
                  drop_rate=drop_rate,
                  expansion=4,
                  **kwargs).build_model()


def _get_weight_name_map(blocks):
    name_map = {
        "stem/conv1/conv2d/kernel:0": "conv1.weight",
        "stem/conv1/batch_norm/gamma:0": "bn1.weight",
        "stem/conv1/batch_norm/beta:0": "bn1.bias",
        "stem/conv1/batch_norm/moving_mean:0": "bn1.running_mean",
        "stem/conv1/batch_norm/moving_variance:0": "bn1.running_var",
    }

    for i in range(1, 5):
        for j in range(blocks[i - 1]):
            for k in range(1, 4):
                n1 = "layer%d/%d/conv%d" % (i, j, k)
                n2 = "layer%d.%d" % (i, j)
                m = {
                    "%s/conv2d/kernel:0" % n1: "%s.conv%d.weight" % (n2, k), 
                    "%s/batch_norm/gamma:0" % n1: "%s.bn%d.weight" % (n2, k),
                    "%s/batch_norm/beta:0" % n1: "%s.bn%d.bias" % (n2, k),
                    "%s/batch_norm/moving_mean:0" % n1: "%s.bn%d.running_mean" % (n2, k),
                    "%s/batch_norm/moving_variance:0" % n1: "%s.bn%d.running_var" % (n2, k),
                    "layer%d/0/shortcut/conv2d/kernel:0" % i: "layer%d.0.downsample.0.weight" % i,
                    "layer%d/0/shortcut/batch_norm/gamma:0" % i: "layer%d.0.downsample.1.weight" % i,
                    "layer%d/0/shortcut/batch_norm/beta:0" % i: "layer%d.0.downsample.1.bias" % i,
                    "layer%d/0/shortcut/batch_norm/moving_mean:0" % i: "layer%d.0.downsample.1.running_mean" % i,
                    "layer%d/0/shortcut/batch_norm/moving_variance:0" % i: "layer%d.0.downsample.1.running_var" % i
                }
                name_map.update(m)
    
    name_map["logits/kernel:0"] = "fc.weight"
    name_map["logits/bias:0"] = "fc.bias"

    return name_map


def _get_weights_from_pretrained(model, pretrained_weights_path, blocks):
    import torch
    import numpy as np

    pretrained = torch.load(pretrained_weights_path, map_location="cpu")
    # for k, v in pretrained.items():
    #     print(k)
    name_map = _get_weight_name_map(blocks)
    
    for w in model.weights:
        name = w.name
        pw = pretrained[name_map[name]].detach().numpy()
        if len(pw.shape) == 4:
            pw = np.transpose(pw, [2, 3, 1, 0])
        if len(pw.shape) == 2:
            pw = np.transpose(pw, [1, 0])
        w.assign(pw)


if __name__ == '__main__':
    # from ..common import fuse
    name = "resnet50"
    block_fn = bottleneck
    blocks = [3, 4, 6, 3]
    resnet = ResNet(name=name,
                    blocks=blocks,
                    block_fn=block_fn,
                    normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True))
    
    model = resnet.build_model()
    model(tf.random.uniform([1, 224, 224, 3]))
    # model.summary()
    # _get_weights_from_pretrained(model, "/home/bail/Downloads/%s.pth" % name, blocks)
    
    # fuse(model, block_fn)

    with tf.io.gfile.GFile("/home/bail/Documents/pandas.jpg", "rb") as gf:
        images = tf.image.decode_jpeg(gf.read())

    images = tf.image.resize(images, (224, 224))
    images = tf.expand_dims(images, axis=0)
    lbl = model(images, training=False)
    top5prob, top5class = tf.nn.top_k(tf.squeeze(tf.nn.softmax(lbl, -1), axis=0), k=5)
    print("prob:", top5prob.numpy())
    print("class:", top5class.numpy())
    
    tf.saved_model.save( model, "./resnet50")
    # model.save_weights("/Users/bailang/Downloads/pretrained_weights/%s.h5" % name)
    # model.save_weights("/Users/bailang/Downloads/pretrained_weights/%s/model.ckpt" % name)
