import tensorflow as tf
from ..common import ConvNormActBlock
from models.builder import BACKBONES
from models.backbones import Backbone
from core.layers import build_activation
from core.layers import build_convolution
from core.layers import build_normalization


def bottleneck(inputs,
               filters, 
               strides=1,
               dilation_rate=1,
               data_format="channels_last",
               normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-3, axis=-1, trainable=True),
               activation=dict(activation="leaky_relu", alpha=0.1),
               kernel_initializer="glorot_uniform",
               trainable=True,
               shortcut=True,
               name="bottleneck"):
    channel_axis = -1 if data_format == "channels_last" else 1
    normalization["axis"] = channel_axis
    if not trainable:
        normalization["trainable"] = trainable
    
    x = ConvNormActBlock(filters=filters,
                         kernel_size=3,
                         strides=strides,
                         dilation_rate=1 if strides == 2 else dilation_rate,
                         normalization=normalization,
                         activation=activation,
                         data_format=data_format,
                         trainable=trainable,
                         kernel_initializer=kernel_initializer,
                         name=name + "/conv1")(inputs)
    x = ConvNormActBlock(filters=filters,
                         kernel_size=3,
                         dilation_rate=dilation_rate,
                         data_format=data_format,
                         normalization=normalization,
                         activation=None,
                         trainable=trainable,
                         kernel_initializer=kernel_initializer,
                         name=name + "/conv2")(x)
    shortcut = inputs
    if tf.keras.backend.int_shape(shortcut)[-1] != filters or strides != 1:
        shortcut = ConvNormActBlock(filters=filters,
                                    kernel_size=1,
                                    strides=strides,
                                    data_format=data_format,
                                    kernel_initializer=kernel_initializer,
                                    trainable=trainable,
                                    activation=None,
                                    normalization=normalization,
                                    name=name + "/skip")(shortcut)        
    x = tf.keras.layers.Add(name=name + "/add")([x, shortcut])
    if isinstance(activation, str):
        x = build_activation(activation=activation, name=name + "/" + activation)(x)
    else:
        x = build_activation(name=name + "/" + activation["activation"], **activation)(x)

    return x


class Hourglass(Backbone):
    def __init__(self,
                 name,
                 n=5,
                 num_stacks=2,
                 blocks=[2, 2, 2, 2, 2, 4],
                 filters=[256, 256, 384, 384, 384, 512],
                 convolution="conv2d",
                 normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                 activation=dict(activation="relu"),
                 output_indices=(-1, ),
                 frozen_stages=(-1,),
                 dropblock=None,
                 num_classes=1000,
                 drop_rate=0.5,
                 input_shape=(224, 224, 3),
                 input_tensor=None,
                 **kwargs):
        super(Hourglass, self).__init__(name=name,
                                        convolution=convolution,
                                        normalization=normalization,
                                        activation=activation,
                                        output_indices=output_indices,
                                        frozen_stages=frozen_stages,
                                        dropblock=dropblock,
                                        num_classes=num_classes,
                                        input_shape=input_shape,
                                        input_tensor=input_tensor,
                                        drop_rate=drop_rate,
                                        **kwargs)
        self.num_stacks = num_stacks
        self.blocks = blocks
        self.filters = filters
        self._n = n
    
    def _make_layer(self, x, num_blocks, infilters, outfilters, name, **kwargs):
        for i in range(num_blocks):
            x = bottleneck(x, 
                           filters=outfilters,
                           normalization=self.normalization,
                           data_format=self.data_format,
                           activation=self.activation,
                           name=name + "/%d" % i,
                           **kwargs)
        
        return x
    
    def _make_layer_revr(self, x, num_blocks, infilters, outfilters, name, **kwargs):
        for i in range(num_blocks - 1):
            x = bottleneck(x, 
                           filters=infilters,
                           normalization=self.normalization,
                           data_format=self.data_format,
                           activation=self.activation,
                           name=name + "/%d" % i,
                           **kwargs)
        x = bottleneck(x, 
                       filters=outfilters,
                       normalization=self.normalization,
                       data_format=self.data_format,
                       activation=self.activation,
                       name=name + "/%d" % (num_blocks - 1),
                       **kwargs)
        
        return x
    
    def _make_hg_layer(self, x, num_blocks, filters, name, **kwargs):
        x = bottleneck(x, 
                       filters=filters,
                       strides=2,
                       normalization=self.normalization,
                       data_format=self.data_format,
                       activation=self.activation,
                       name=name + "/0",
                       **kwargs)
        for i in range(num_blocks - 1):
            x = bottleneck(x, 
                           filters=filters,
                           normalization=self.normalization,
                           data_format=self.data_format,
                           activation=self.activation,
                           name=name + "/%d" % (i + 1),
                           **kwargs)
        
        return x
    
    def stack_module(self, x, n, filters, blocks, name, **kwargs):
        self.n = n
        curr_blocks = blocks[0]
        next_blocks = blocks[1]

        curr_filters = filters[0]
        next_filters = filters[1]

        up1 = self._make_layer(x, curr_blocks, curr_filters, curr_filters, name + "/up1", **kwargs)
        self.prev_filters = curr_filters
        low1 = self._make_hg_layer(x, curr_blocks, next_filters, name=name + "/low1", **kwargs)
        if self.n > 1:
            low2 = self.stack_module(low1, n - 1, filters[1:], blocks[1:], name=name + "/low2", **kwargs)
        else:
            low2 = self._make_layer(low1, next_blocks, next_filters, next_filters, name + "/low2", **kwargs)
        
        low3 = self._make_layer_revr(low2, curr_blocks, next_filters, curr_filters, name=name + "/low3", **kwargs)
        up2 = tf.keras.layers.UpSampling2D(data_format=self.data_format)(low3)

        return tf.keras.layers.Add()([up1, up2])
    
    def build_model(self):
        trainable = 0 not in self.frozen_stages
        
        def _fn(inp):
            mean = tf.constant([[[[0.408, 0.447, 0.470]]]], inp.dtype) * 255.
            std = 1. / (tf.constant([[[[0.289, 0.274, 0.278]]]], inp.dtype) * 255.)
            x = (inp - mean) * std
            
            return x

        curr_filters = self.filters[0]
        kernel_initializer="he_normal"
        inputs = tf.keras.layers.Lambda(function=_fn, name="norm_input")(self.img_input)
        x = ConvNormActBlock(filters=128,
                             kernel_size=7,
                             strides=2,
                             normalization=self.normalization,
                             activation=self.activation,
                             trainable=trainable,
                             data_format=self.data_format,
                             kernel_initializer=kernel_initializer,
                             name="pre/0")(inputs)
        inter = bottleneck(inputs=x,
                           filters=256,
                           strides=2,
                           data_format=self.data_format,
                           kernel_initializer=kernel_initializer,
                           normalization=self.normalization,
                           activation=self.activation,
                           trainable=1 not in self.frozen_stages,
                           name="pre/1")
        
        self.prev_filters = 256
        outputs = []
        for i in range(self.num_stacks):
            kp = self.stack_module(inter, self._n, self.filters, self.blocks, name="kps/%d" % i)
            cnv = ConvNormActBlock(filters=curr_filters,
                                   kernel_size=3,
                                   data_format=self.data_format,
                                   normalization=self.normalization,
                                   activation=self.activation,
                                   name="cnvs/%d" % i)(kp)
                                   
            outputs.append(cnv)
            if i < self.num_stacks - 1:
                x1 = ConvNormActBlock(filters=curr_filters,
                                      kernel_size=1,
                                      data_format=self.data_format,
                                      normalization=self.normalization,
                                      activation=None,
                                      name="inters_/%d" % i)(inter)
                x2 = ConvNormActBlock(filters=curr_filters,
                                      kernel_size=1,
                                      data_format=self.data_format,
                                      normalization=self.normalization,
                                      activation=None,
                                      name="cnvs_/%d" % i)(cnv)
                inter = tf.keras.layers.Add()([x1, x2])
                inter = build_activation(**self.activation)(inter)
                inter = bottleneck(inter, 
                                   filters=curr_filters,
                                   normalization=self.normalization,
                                   data_format=self.data_format,
                                   activation=self.activation,
                                   name="inters/%d" % i)
        
        if -1 not in self.output_indices:
            outputs = [o for i, o in enumerate(outputs) if i + 1 in self.output_indices]
        return tf.keras.Model(inputs=self.img_input, outputs=outputs, name=self.name)


@BACKBONES.register("HourglassNet")
def HourglassNet(normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                 activation=dict(activation="relu"),
                 frozen_stages=(-1,),
                 dropblock=None,
                 input_shape=(512, 512, 3),
                 **kwargs):
    return Hourglass("hourglass_net",
                     n=5,
                     num_stacks=2,
                     blocks=[2, 2, 2, 2, 2, 4],
                     filters=[256, 256, 384, 384, 384, 512],
                     normalization=normalization,
                     activation=activation,
                     frozen_stages=frozen_stages,
                     dropblock=dropblock,
                     input_shape=input_shape,
                     **kwargs).build_model()


def _load_weight_from_torch(model, torch_weights_path="/home/bail/Downloads/ctdet_coco_hg.pth"):
    import torch
    import numpy as np

    loaded = torch.load(torch_weights_path)["state_dict"]

    for weight in model.weights:
        name = weight.name.split(":")[0]
        name = name.replace("/", ".")
        if "batch_norm" in name:
            name = name.replace("batch_norm", "bn")
        if "kernel" in name:
            name = name.replace("kernel", "weight")
        if "gamma" in name:
            name = name.replace("gamma", "weight")
        if "beta" in name:
            name = name.replace("beta", "bias")
        if "moving_mean" in name:
            name = name.replace("moving_mean", "running_mean")
        if "moving_variance" in name:
            name = name.replace("moving_variance", "running_var")
        if "conv2d.weight" in name:
            name = name.replace("conv2d.weight", "conv.weight")
        if "skip.bn" in name:
            name = name.replace("skip.bn", "skip.1")
        if "skip.conv" in name:
            name = name.replace("skip.conv", "skip.0")
        if "cnvs_.0.bn" in name:
            name = name.replace("cnvs_.0.bn", "cnvs_.0.1")
        if "cnvs_.0.conv" in name:
            name = name.replace("cnvs_.0.conv", "cnvs_.0.0")
        if "inters_.0.bn" in name:
            name = name.replace("inters_.0.bn", "inters_.0.1")
        if "inters_.0.conv" in name:
            name = name.replace("inters_.0.conv", "inters_.0.0")
        if "conv1.conv" in name:
            name = name.replace("conv1.conv", "conv1")
        if "conv1.bn" in name:
            name = name.replace("conv1.bn", "bn1")
        if "conv2.conv" in name:
            name = name.replace("conv2.conv", "conv2")
        if "conv2.bn" in name:
            name = name.replace("conv2.bn", "bn2")
        
        name = "module." + name
        
        tw = loaded[name].numpy()
        if len(tw.shape) == 4:
            tw = np.transpose(tw, (2, 3, 1, 0))
        
        if len(tw.shape) == 2:
            tw = np.transpose(tw, (1, 0))
        weight.assign(tw)

if __name__ == "__main__":
    net = HourglassNet()

    _load_weight_from_torch(net)

        

