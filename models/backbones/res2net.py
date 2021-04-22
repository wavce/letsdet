import math
import tensorflow  as tf 
from .backbone import Backbone
from ..builder import BACKBONES
from ..common import ConvNormActBlock
from core.layers import build_activation


def bottle2neck(inputs,
                filters,
                strides=1,
                scale=4,
                base_width=26,
                dilation_rate=1,
                data_format="channels_last",
                normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                activation=dict(activation="relu"),
                downsample=False,
                trainable=True,
                dropblock=None,
                stype="normal",
                expansion=4,
                name="Bottle2neck"):
    width = int(math.floor(filters * (base_width / 64.)))
    channel_axis = -1 if data_format == "channels_last" else 1

    x = ConvNormActBlock(filters=width * scale,
                         kernel_size=1,
                         trainable=trainable,
                         data_format=data_format,
                         normalization=normalization,
                         activation=activation,
                         dropblock=dropblock,
                         name=name + "/conv1")(inputs)
    
    num_convs = scale if scale == 1 else scale - 1
    spx = tf.keras.layers.Lambda(lambda inp: tf.split(inp, scale, channel_axis), name=name + "/split")(x)
    for i in range(num_convs):
        if i == 0 or stype == "stage":
            sp = spx[i]
        else:
            sp = tf.keras.layers.Add(name=name + "/add%d" % i)([sp, spx[i]])
        sp = ConvNormActBlock(filters=width, 
                              kernel_size=3, 
                              strides=strides, 
                              data_format=data_format, 
                              dilation_rate=dilation_rate if strides == 1 else 1, 
                              trainable=trainable,
                              normalization=normalization,
                              activation=activation,
                              dropblock=dropblock,
                              name=name + "/convs/%d" % i)(sp)
        if i == 0:
            x = sp
        else:
            x = tf.keras.layers.Concatenate(channel_axis, name=name + "/cat%d" % i)([x, sp])
    if scale != 1 and stype == "normal":
        x = tf.keras.layers.Concatenate(channel_axis, name=name + "/cat%d" % num_convs)([x, spx[num_convs]])
    elif scale != 1 and stype == "stage":
        padding = "same"
        sp_ = spx[num_convs]
        if strides != 1:
            sp_ = tf.keras.layers.ZeroPadding2D(((1, 1), (1, 1)), data_format=self.data_format, name=name + "/pad")(sp_)
            padding = "valid"
            
        sp_ = tf.keras.layers.AvgPool2D(3, strides, padding, data_format, name=name + "/avgpool")(sp_)
        x = tf.keras.layers.Concatenate(channel_axis, name=name + "/cat%d" % num_convs)([x, sp_])
        
    x = ConvNormActBlock(filters=filters * expansion,
                         kernel_size=1,
                         trainable=trainable,
                         data_format=data_format,
                         normalization=normalization,
                         activation=None,
                         dropblock=dropblock,
                         name=name + "/conv3")(x)

    shortcut = inputs
    if downsample:
        shortcut = ConvNormActBlock(filters=filters * expansion,
                                    kernel_size=1,
                                    strides=strides,
                                    trainable=trainable,
                                    data_format=data_format,
                                    normalization=normalization,
                                    activation=None,
                                    dropblock=dropblock,
                                    name=name + "/downsample")(shortcut)
    x = tf.keras.layers.Add(name=name + "/add")([x, shortcut])
    x = build_activation(**activation, name=name + "/" + activation["activation"])(x)
    
    return x


class Res2Net(Backbone):
    def __init__(self, 
                 name, 
                 num_blocks,
                 dropblock=dict(block_size=7, drop_rate=0.1),
                 normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                 activation=dict(activation="relu"),
                 output_indices=(3, 4), 
                 strides=(2, 2, 2, 2, 2), 
                 dilation_rates=(1, 1, 1, 1, 1), 
                 frozen_stages=(-1, ), 
                 input_shape=None, 
                 input_tensor=None, 
                 base_width=26,
                 scale=4,
                 num_classes=1000, 
                 drop_rate=0.5,
                 **kwargs):
        super(Res2Net, self).__init__(name, 
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
        self.num_blocks = num_blocks
        self.base_width = base_width
        self.scale = scale

    def build_model(self):
        def _norm(inp):
            mean = tf.constant([0.485, 0.456, 0.406], inp.dtype, [1, 1, 1, 3]) * 255.
            std = 1. / (tf.constant([0.229, 0.224, 0.225], inp.dtype, [1, 1, 1, 3]) * 255.)

            return (inp - mean) * std 

        x = tf.keras.layers.Lambda(_norm, name="norm_input")(self.img_input) 
        self.infilters = 64
        x = ConvNormActBlock(filters=self.infilters, 
                             kernel_size=7, 
                             strides=self.strides[0],
                             dilation_rate=self.dilation_rates[0],
                             data_format=self.data_format,
                             normalization=self.normalization,
                             activation=self.activation,
                             dropblock=self.dropblock,
                             trainable=1 not in self.frozen_stages,
                             name="conv1")(x)
        outputs = [x]
        x = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)), data_format=self.data_format)(x)
        x = tf.keras.layers.MaxPool2D((3, 3), self.strides[1], "valid", self.data_format, name="maxpool")(x)
        x = self._make_layer(x, 64,  self.num_blocks[0], 1,               self.dilation_rates[1], 2 not in self.frozen_stages, name="layer1")
        outputs.append(x)
        x = self._make_layer(x, 128, self.num_blocks[1], self.strides[2], self.dilation_rates[2], 3 not in self.frozen_stages, name="layer2")
        outputs.append(x)
        x = self._make_layer(x, 256, self.num_blocks[2], self.strides[3], self.dilation_rates[3], 4 not in self.frozen_stages, name="layer3")
        outputs.append(x)
        x = self._make_layer(x, 512, self.num_blocks[3], self.strides[4], self.dilation_rates[4], 5 not in self.frozen_stages, name="layer4")
        outputs.append(x)

        if -1 not in self.output_indices:
            outputs = (outputs[i-1] for i in self.output_indices)
        else:
            x = tf.keras.layers.GlobalAvgPool2D(data_format=self.data_format, name="gloabl_avgpool")(x)
            if self.drop_rate and self.drop_rate > 0.:
                x = tf.keras.layers.Dropout(rate=self.drop_rate, name="drop")(x)
            outputs = tf.keras.layers.Dense(units=self.num_classes, name="logits")(x)

        return tf.keras.Model(inputs=self.img_input, outputs=outputs, name=self.name)
    
    def _make_layer(self, inputs, filters, num_block, strides=1, dilation_rate=1, trainable=True, name="layer"):
        x = bottle2neck(inputs,
                        filters,
                        strides=strides,
                        base_width=self.base_width,
                        scale=self.scale,
                        dilation_rate=dilation_rate,
                        data_format=self.data_format,
                        trainable=trainable,
                        dropblock=self.dropblock,
                        normalization=self.normalization,
                        activation=self.activation,
                        downsample=strides != 1 or self.infilters != filters * 4,
                        stype="stage",
                        name=name + "/0")
        
        for i in range(1, num_block):
            x = bottle2neck(x,
                            filters,
                            strides=1,
                            base_width=self.base_width,
                            scale=self.scale,
                            dilation_rate=dilation_rate,
                            data_format=self.data_format,
                            trainable=trainable,
                            dropblock=self.dropblock,
                            normalization=self.normalization,
                            activation=self.activation,
                            downsample=False,
                            name=name + "/%d" % i)
        self.infilters = filters * 4

        return x

    
@BACKBONES.register("Res2Net50_26W4S")
def Res2Net50_26W4S(dropblock=None, 
                    normalization=dict(normalization='batch_norm', momentum=0.9, epsilon=1e-05, axis = -1, trainable =True), 
                    activation=dict(activation='relu'), 
                    output_indices=(3, 4), 
                    strides=(2, 2, 2, 2, 2), 
                    dilation_rates=(1, 1, 1, 1, 1), 
                    frozen_stages=(-1, ), 
                    input_shape=None, 
                    input_tensor=None,
                    num_classes=1000, 
                    drop_rate=0.5,
                    **kwargs):
    return Res2Net("res2net50_26w4s", 
                   num_blocks=[3, 4, 6, 3],
                   dropblock=dropblock, 
                   normalization=normalization, 
                   activation=activation, 
                   output_indices=output_indices, 
                   strides=strides, 
                   dilation_rates=dilation_rates, 
                   frozen_stages=frozen_stages, 
                   input_shape=input_shape, 
                   input_tensor=input_tensor, 
                   base_width=26, 
                   scale=4, 
                   num_classes=num_classes, 
                   drop_rate=drop_rate,
                   **kwargs).build_model()


@BACKBONES.register("Res2Net101_26W4S")
def Res2Net101_26W4S(dropblock=None, 
                     normalization=dict(normalization='batch_norm', momentum=0.9, epsilon=1e-05, axis = -1, trainable =True), 
                     activation=dict(activation='relu'), 
                     output_indices=(3, 4), 
                     strides=(2, 2, 2, 2, 2), 
                     dilation_rates=(1, 1, 1, 1, 1), 
                     frozen_stages=(-1, ), 
                     input_shape=None, 
                     input_tensor=None,
                     num_classes=1000, 
                     drop_rate=0.5,
                     **kwargs):
    return Res2Net("res2net101_26w4s", 
                   num_blocks=[3, 4, 23, 3],
                   dropblock=dropblock, 
                   normalization=normalization, 
                   activation=activation, 
                   output_indices=output_indices, 
                   strides=strides, 
                   dilation_rates=dilation_rates, 
                   frozen_stages=frozen_stages, 
                   input_shape=input_shape, 
                   input_tensor=input_tensor, 
                   base_width=26, 
                   scale=4, 
                   num_classes=num_classes, 
                   drop_rate=drop_rate,
                   **kwargs).build_model()


@BACKBONES.register("Res2Net50_26W6S")
def Res2Net50_26W6S(dropblock=None, 
                    normalization=dict(normalization='batch_norm', momentum=0.9, epsilon=1e-05, axis = -1, trainable =True), 
                    activation=dict(activation='relu'), 
                    output_indices=(-1, ), 
                    strides=(2, 2, 2, 2, 2), 
                    dilation_rates=(1, 1, 1, 1, 1), 
                    frozen_stages=(-1, ), 
                    input_shape=None, 
                    input_tensor=None,
                    num_classes=1000, 
                    drop_rate=0.5,
                    **kwargs):
    return Res2Net("res2net50_26w6s", 
                   num_blocks=[3, 4, 6, 3],
                   dropblock=dropblock, 
                   normalization=normalization, 
                   activation=activation, 
                   output_indices=output_indices, 
                   strides=strides, 
                   dilation_rates=dilation_rates, 
                   frozen_stages=frozen_stages, 
                   input_shape=input_shape, 
                   input_tensor=input_tensor, 
                   base_width=26, 
                   scale=6, 
                   num_classes=num_classes, 
                   drop_rate=drop_rate,
                   **kwargs).build_model()


@BACKBONES.register("Res2Net50_26W8S")
def Res2Net50_26W8S(dropblock=None, 
                    normalization=dict(normalization='batch_norm', momentum=0.9, epsilon=1e-05, axis=-1, trainable =True), 
                    activation=dict(activation='relu'), 
                    output_indices=(3, 4), 
                    strides=(2, 2, 2, 2, 2), 
                    dilation_rates=(1, 1, 1, 1, 1), 
                    frozen_stages=(-1, ), 
                    input_shape=None, 
                    input_tensor=None,
                    num_classes=1000, 
                    drop_rate=0.5,
                    **kwargs):
    return Res2Net("res2net50_26w8s", 
                   num_blocks=[3, 4, 6, 3],
                   dropblock=dropblock, 
                   normalization=normalization, 
                   activation=activation, 
                   output_indices=output_indices, 
                   strides=strides, 
                   dilation_rates=dilation_rates, 
                   frozen_stages=frozen_stages, 
                   input_shape=input_shape, 
                   input_tensor=input_tensor, 
                   base_width=26, 
                   scale=8, 
                   num_classes=num_classes, 
                   drop_rate=drop_rate,
                   **kwargs).build_model()


@BACKBONES.register("Res2Net50_48W2S")
def Res2Net50_48W2S(dropblock=None, 
                    normalization=dict(normalization='batch_norm', momentum=0.9, epsilon=1e-05, axis = -1, trainable =True), 
                    activation=dict(activation='relu'), 
                    output_indices=(3, 4), 
                    strides=(2, 2, 2, 2, 2), 
                    dilation_rates=(1, 1, 1, 1, 1), 
                    frozen_stages=(-1, ), 
                    input_shape=None, 
                    input_tensor=None,
                    num_classes=1000, 
                    drop_rate=0.5,
                    **kwargs):
    return Res2Net("res2net50_48w2s", 
                   num_blocks=[3, 4, 6, 3],
                   dropblock=dropblock, 
                   normalization=normalization, 
                   activation=activation, 
                   output_indices=output_indices, 
                   strides=strides, 
                   dilation_rates=dilation_rates, 
                   frozen_stages=frozen_stages, 
                   input_shape=input_shape, 
                   input_tensor=input_tensor, 
                   base_width=48, 
                   scale=2, 
                   num_classes=num_classes, 
                   drop_rate=drop_rate,
                   **kwargs).build_model()


@BACKBONES.register("Res2Net50_14W8S")
def Res2Net50_14W8S(dropblock=None, 
                    normalization=dict(normalization='batch_norm', momentum=0.9, epsilon=1e-05, axis = -1, trainable =True), 
                    activation=dict(activation='relu'), 
                    output_indices=(3, 4), 
                    strides=(2, 2, 2, 2, 2), 
                    dilation_rates=(1, 1, 1, 1, 1), 
                    frozen_stages=(-1, ), 
                    input_shape=None, 
                    input_tensor=None,
                    num_classes=1000, 
                    drop_rate=0.5,
                    **kwargs):
    return Res2Net("res2net50_14w8s", 
                   num_blocks=[3, 4, 6, 3],
                   dropblock=dropblock, 
                   normalization=normalization, 
                   activation=activation, 
                   output_indices=output_indices, 
                   strides=strides, 
                   dilation_rates=dilation_rates, 
                   frozen_stages=frozen_stages, 
                   input_shape=input_shape, 
                   input_tensor=input_tensor, 
                   base_width=14, 
                   scale=8, 
                   num_classes=num_classes, 
                   drop_rate=drop_rate,
                   **kwargs).build_model()


def _get_weight_name_map(blocks, scale):
    name_map = {
        "conv1/conv2d/kernel:0": "conv1.weight",
        "conv1/batch_norm/gamma:0": "bn1.weight",
        "conv1/batch_norm/beta:0": "bn1.bias",
        "conv1/batch_norm/moving_mean:0": "bn1.running_mean",
        "conv1/batch_norm/moving_variance:0": "bn1.running_var"
    }

    for i in range(1, 5):
        for j in range(blocks[i - 1]):
            for k in range(1, 4):
                n1 = "layer%d/%d/conv%d" % (i, j, k)
                n2 = "layer%d.%d" % (i, j)
                if k != 2:
                    m = {
                        "%s/conv2d/kernel:0" % n1: "%s.conv%d.weight" % (n2, k), 
                        "%s/batch_norm/gamma:0" % n1: "%s.bn%d.weight" % (n2, k),
                        "%s/batch_norm/beta:0" % n1: "%s.bn%d.bias" % (n2, k),
                        "%s/batch_norm/moving_mean:0" % n1: "%s.bn%d.running_mean" % (n2, k),
                        "%s/batch_norm/moving_variance:0" % n1: "%s.bn%d.running_var" % (n2, k),
                        "layer%d/0/downsample/conv2d/kernel:0" % i: "layer%d.0.downsample.0.weight" % i,
                        "layer%d/0/downsample/batch_norm/gamma:0" % i: "layer%d.0.downsample.1.weight" % i,
                        "layer%d/0/downsample/batch_norm/beta:0" % i: "layer%d.0.downsample.1.bias" % i,
                        "layer%d/0/downsample/batch_norm/moving_mean:0" % i: "layer%d.0.downsample.1.running_mean" % i,
                        "layer%d/0/downsample/batch_norm/moving_variance:0" % i: "layer%d.0.downsample.1.running_var" % i
                    }
                    name_map.update(m)
                else:
                    for s in range(scale - 1):
                        m = {
                            "layer%d/%d/convs/%d/conv2d/kernel:0" % (i, j, s): "%s.convs.%d.weight" % (n2, s), 
                            "layer%d/%d/convs/%d/batch_norm/gamma:0" % (i, j, s): "%s.bns.%d.weight" % (n2, s),
                            "layer%d/%d/convs/%d/batch_norm/beta:0" % (i, j, s): "%s.bns.%d.bias" % (n2, s),
                            "layer%d/%d/convs/%d/batch_norm/moving_mean:0" % (i, j, s): "%s.bns.%d.running_mean" % (n2, s),
                            "layer%d/%d/convs/%d/batch_norm/moving_variance:0" % (i, j, s): "%s.bns.%d.running_var" % (n2, s),
                        }
                        name_map.update(m)
    
    name_map["logits/kernel:0"] = "fc.weight"
    name_map["logits/bias:0"] = "fc.bias"

    return name_map


def _torch2h5(model, torch_weight_path, blocks, scale):
    import torch
    import numpy as np

    net = torch.load(torch_weight_path, map_location=torch.device('cpu'))
    
    # for k, _ in net.items():
    #     if "tracked" in k:
    #         continue
    #     print(k) 

    name_map = _get_weight_name_map(blocks, scale)
    for weight in model.weights:
        name = weight.name
        
        tw = net[name_map[name]].numpy()
        if len(tw.shape) == 4:
            tw = np.transpose(tw, (2, 3, 1, 0))
        if len(tw.shape) == 2:
            tw = np.transpose(tw, (1, 0))
        
        weight.assign(tw)

    del net


if __name__ == "__main__":
    name = "res2net101_26w_4s"
    blocks = [3, 4, 23, 3]
    scale = 4
    model = Res2Net101_26W4S(input_shape=(224, 224, 3), output_indices=(-1, ))
    # model(tf.random.uniform([1, 224, 224, 3], 0, 255))
    # model.summary()
    _torch2h5(model, "/Users/bailang/Downloads/pretrained_weights/%s.pth" % name, blocks, scale)

    with tf.io.gfile.GFile("/Users/bailang/Documents/pandas.jpg", "rb") as gf:
        images = tf.image.decode_jpeg(gf.read())

    images = tf.image.resize(images, (224, 224))[None]
    lbl = model(images, training=False)
    
    top5prob, top5class = tf.nn.top_k(tf.squeeze(tf.nn.softmax(lbl, -1), axis=0), k=5)
    print("prob:", top5prob.numpy())
    print("class:", top5class.numpy())
    
    model.save_weights("/Users/bailang/Downloads/pretrained_weights/%s.h5" % name)
    model.save_weights("/Users/bailang/Downloads/pretrained_weights/%s/model.ckpt" % name)


