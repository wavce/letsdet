import tensorflow  as tf 
from .backbone import Backbone
from ..builder import BACKBONES
from core.layers import DropBlock2D
from core.layers import build_activation
from core.layers import build_normalization


class BasicBlock(tf.keras.Model):
    expansion = 1

    def __init__(self,
                 filters, 
                 in_filters, 
                 strides=1, 
                 dilation_rate=1,
                 data_format="channels_last",
                 dropblock=dict(block_size=7, drop_rate=0.1),
                 normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                 activation=dict(activation="relu"),
                 trainable=True,
                 name="basic_block",
                 **kwargs):
        super(BasicBlock, self).__init__(name=name, **kwargs)
        norm = normalization["normalization"]
        act = activation["activation"]
        normalization = normalization.copy()
        normalization["tranable"] = trainable
        self.conv1 = tf.keras.layers.Conv2D(filters, 
                                            kernel_size=(3, 3),
                                            strides=strides,
                                            padding="same",
                                            data_format=data_format,
                                            dilation_rate=dilation_rate,
                                            use_bias=False,
                                            name="conv1")
        self.norm1 = build_normalization(**normalization, name="%s1" % norm)
        if dropblock is not None:
            self.dropblock1 = DropBlock2D(**dropblock, data_format=data_format, name="dropblock1")
        self.act1 = build_activation(**activation, name="%s1" % act)

        self.conv2 = tf.keras.layers.Conv2D(filters, 
                                            kernel_size=(3, 3),
                                            padding="same",
                                            data_format=data_format,
                                            dilation_rate=dilation_rate,
                                            use_bias=False,
                                            name="conv2")
        self.norm2 = build_normalization(**normalization, name="%s2" % norm)
        if dropblock is not None:
            self.dropblock2 = DropBlock2D(**dropblock, data_format=data_format, name="dropblock2")
        self.act2 = build_activation(**activation, name="/%s2" % act)

        self.downsample = None
        if strides != 1 or in_filters != filters * BasicBlock.expansion:
            self.downsample = tf.keras.Sequential([
                tf.keras.layers.Conv2D(filters, 
                                    kernel_size=(1, 1),
                                    strides=strides,
                                    data_format=data_format,
                                    use_bias=False,
                                    name="0"),
                build_normalization(**normalization, name="1")
            ], name=name + "/downsample")
            if dropblock is not None:
                self.dropblock3 = DropBlock2D(**dropblock, data_format=data_format, name="downsample/dropblock")
        
        self.act2 = build_activation(**activation, name="%s2" % act)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.norm1(x)
        if hasattr(self, "dropblock1"):
            x = self.dropblock1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        if hasattr(self, "dropblock2"):
            x = self.dropblock2(x)

        shortcut = inputs
        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
            if hasattr(self, "dropblock3"):
                shortcut = self.dropblock3(shortcut)
        x = self.act2(x + shortcut)

        return x


class Bottleneck(tf.keras.Model):
    expansion=4

    def __init__(self,
                 filters, 
                 in_filters, 
                 strides=1, 
                 dilation_rate=1,
                 data_format="channels_last",
                 dropblock=dict(block_size=7, drop_rate=0.1),
                 normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                 activation=dict(activation="relu"),
                 trainable=True,
                 name="bottleneck"):
        super(Bottleneck, self).__init__(name=name)
        norm = normalization["normalization"]
        act = activation["activation"]
        normalization = normalization.copy()
        normalization["tranable"] = trainable
        
        self.conv1 = tf.keras.layers.Conv2D(filters, 
                                            kernel_size=1,
                                            data_format=data_format,
                                            padding="same",
                                            use_bias=False,
                                            name="conv1")
        self.norm1 = build_normalization(**normalization, name="%s1" % norm)
        if dropblock is not None:
            self.dropblock1 = DropBlock2D(**dropblock, data_format=data_format, name="dropblock1")
        self.act1 = build_activation(**activation, name="%s1" % act)

        self.conv2 = tf.keras.layers.Conv2D(filters, 
                                            kernel_size=(3, 3),
                                            strides=strides,
                                            padding="same",
                                            data_format=data_format,
                                            dilation_rate=dilation_rate,
                                            use_bias=False,
                                            name="conv2")
        self.norm2 = build_normalization(**normalization, name="%s2" % norm)
        if dropblock is not None:
            self.dropblock2 = DropBlock2D(**dropblock, data_format=data_format, name="dropblock2")
        self.act2 = build_activation(**activation, name="%s2" % act)
        self.conv3 = tf.keras.layers.Conv2D(filters * Bottleneck.expansion, 
                                            kernel_size=1,
                                            padding="same",
                                            data_format=data_format,
                                            use_bias=False,
                                            name="conv3")
        self.norm3 = build_normalization(**normalization, name="%s3" % norm)
        if dropblock is not None:
            self.dropblock3 = DropBlock2D(**dropblock, data_format=data_format, name="dropblock3")

        self.downsample = None
        if strides != 1 or in_filters != filters * Bottleneck.expansion:
            self.downsample = tf.keras.Sequential([
                tf.keras.layers.Conv2D(filters * Bottleneck.expansion, 
                                       kernel_size=(1, 1),
                                       strides=strides,
                                       padding="same",
                                       data_format=data_format,
                                       use_bias=False,
                                       name="0"),
                build_normalization(**normalization, name="1")
            ], name="downsample")
            if dropblock is not None:
                self.dropblock4 = DropBlock2D(
                    **dropblock, data_format=data_format, name="downsample/dropblock")
        
        self.sum = tf.keras.layers.Add(name="sum")
        self.act3 = build_activation(**activation, name="%s3" % act)

    def call(self, inputs, training=None):
        shortcut = inputs

        x = self.conv1(inputs)
        x = self.norm1(x, training=training)
        if hasattr(self, "dropblock1"):
            x = self.dropblock1(x, training=training)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.norm2(x, training=training)
        if hasattr(self, "dropblock2"):
            x = self.dropblock2(x, training=training)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.norm3(x, training=training)
        if hasattr(self, "dropblock3"):
            x = self.dropblock3(x, training=training)

        if self.downsample is not None:
            shortcut = self.downsample(shortcut, training=training)
            if hasattr(self, "dropblock4"):
                shortcut = self.dropblock4(shortcut, training=training)

        x = self.sum([shortcut, x])
        x = self.act3(x)

        return x


BLOCKS = dict(BASIC=BasicBlock, BOTTLENECK=Bottleneck)


CFGS = dict(
    hrnet_w18=dict(
        stage1=dict(num_modules=1, num_branches=1, block='BOTTLENECK', num_blocks=(4, ), filters=(64, ), fuse_method='SUM'),
        stage2=dict(num_modules=1, num_branches=2, block='BASIC', num_blocks=(4, 4), filters=(18, 36), fuse_method='SUM'),
        stage3=dict(num_modules=4, num_branches=3, block='BASIC', num_blocks=(4, 4, 4), filters=(18, 36, 72), fuse_method='SUM'),
        stage4=dict(num_modules=3, num_branches=4, block='BASIC', num_blocks=(4, 4, 4, 4), filters=(18, 36, 72, 144), fuse_method='SUM')),
    )


class HighResolutionModule(tf.keras.Model):
    def __init__(self, 
                 num_branches, 
                 block_fn, 
                 num_blocks, 
                 filters, 
                 in_filters, 
                 data_format="channels_last",
                 dropblock=dict(block_size=7, drop_rate=0.1),
                 normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                 activation=dict(activation="relu"),
                 fuse_method="SUM", 
                 trainable=True,
                 multi_scale_output=True, 
                 **kwargs):
        super(HighResolutionModule, self).__init__(trainable=trainable, **kwargs)

        self.num_branches = num_branches
        self.block_fn = block_fn
        self.num_blocks = num_blocks
        self.filters = filters
        self.in_filters = in_filters
        self.fuse_method = fuse_method
        self.multi_scale_output = multi_scale_output

        self.data_format = data_format
        self.normalization = normalization
        self.activation = activation
        self.dropblock = dropblock

        self._check_branches()
        self.branches = self._make_branches()
        self.fuse_layers = self._make_fuse_layers()
        self.act = build_activation(**activation)
    
    def _check_branches(self):
        if self.num_branches != len(self.num_blocks):
            error_msg = "num_branches({}) should equal to num_blocks({}).".format(self.num_branches, self.num_blocks)
            raise ValueError(error_msg)

        if self.num_branches != len(self.filters):
            error_msg = "num_branches({}) should equal to filters({}).".format(self.num_branches, self.filters)
            raise ValueError(error_msg)
            
        if self.num_branches != len(self.in_filters):
            error_msg = "num_branches({}) should equal to in_filters({}).".format(self.num_branches, self.in_filters)
            raise ValueError(error_msg)
    
    def _make_one_branch(self, branch_index,  num_blocks, strides=1, dilation_rate=1, name="branches"):
        layers = tf.keras.Sequential(name=name)

        layer = self.block_fn(self.filters[branch_index], 
                              self.in_filters[branch_index], 
                              strides=strides, 
                              dilation_rate=1,
                              data_format=self.data_format,
                              dropblock=self.dropblock,
                              normalization=self.normalization,
                              activation=self.activation,
                              name="0")
        layers.add(layer)
        in_filters = self.filters[branch_index] * self.block_fn.expansion
        for i in range(1, num_blocks):
            layer = self.block_fn(self.filters[branch_index], 
                                  in_filters, 
                                  strides=1, 
                                  dilation_rate=dilation_rate,
                                  data_format=self.data_format,
                                  dropblock=self.dropblock,
                                  normalization=self.normalization,
                                  activation=self.activation,
                                  name="%d" % i)
            layers.add(layer)
        
        return layers
    
    def _make_branches(self):
        branches = []
        
        for i in range(self.num_branches):
            branches.append(self._make_one_branch(i, self.num_blocks[i], name="branches/%d" % i))
        
        return branches

    def _make_fuse_layers(self, name="fuse_layers"):
        if self.num_branches == 1:
            return None
        
        fuse_layers = []
        for i in range(self.num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(self.num_branches):
                if j > i:
                    layer = tf.keras.Sequential([
                        tf.keras.layers.Conv2D(self.in_filters[i], 1, 1, 
                                               padding="same",
                                               data_format=self.data_format, 
                                               use_bias=False,
                                               name="0"),
                        build_normalization(**self.normalization, name="1"),
                        tf.keras.layers.UpSampling2D(2**(j-i), self.data_format, name="2")
                    ], name=name + "/%d/%d" % (i, j))
                    fuse_layer.append(layer)
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            out_filters = self.in_filters[i]
                            layer = tf.keras.Sequential([
                                tf.keras.layers.Conv2D(out_filters, 3, 2, 
                                                       padding="same",
                                                       data_format=self.data_format, 
                                                       use_bias=False,
                                                       name="0"),
                                build_normalization(**self.normalization, name="1")
                            ], name=str(k))
                        else:
                            out_filters = self.in_filters[j]
                            layer = tf.keras.Sequential([
                                tf.keras.layers.Conv2D(out_filters, 3, 2, 
                                                       data_format=self.data_format, 
                                                       padding="same",
                                                       use_bias=False,
                                                       name="0"),
                                build_normalization(**self.normalization, name="1"),
                                build_activation(**self.activation, name="2")
                            ], name=str(k))
                        conv3x3s.append(layer)

                    fuse_layer.append(tf.keras.Sequential(conv3x3s, name=name + "/%d/%d" % (i, j)))
            fuse_layers.append(fuse_layer)
        
        return fuse_layers

    def call(self, inputs, training=None):
        if self.num_branches == 1:
            return [self.branches[0](inputs[0], training=training)]
        
        for i in range(self.num_branches):
            inputs[i] = self.branches[i](inputs[i])
        
        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = inputs[0] if i == 0 else self.fuse_layers[i][0](inputs[0], training=training)
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + inputs[j]
                else:
                    y = y + self.fuse_layers[i][j](inputs[j], training=training)

            x_fuse.append(self.act(y))
        
        return x_fuse

        
class HighResolutionNet(Backbone):
    def __init__(self, 
                 name, 
                 cfg,
                 convolution='conv2d', 
                 normalization=dict(), 
                 activation=dict(), 
                 output_indices=(-1, ), 
                 strides=(2, 2, 2, 2, 2), 
                 dilation_rates=(1, 1, 1, 1, 1), 
                 frozen_stages=(-1, ), 
                 input_shape=None, 
                 input_tensor=None, 
                 dropblock=None, 
                 num_classes=1000, 
                 drop_rate=0.5):
        super(HighResolutionNet, self).__init__(name, 
                                                convolution=convolution, 
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
                                                drop_rate=drop_rate)
    
        self.cfg = cfg
        self._channel_axis = -1 if self.data_format == "channels_last" else 1

    def _make_transition_layer(self, inputs, filters, trainable=True, name="transition"):
        num_filters_pre_layer = [tf.keras.backend.int_shape(inp)[self._channel_axis] for inp in inputs]
        num_branches_cur = len(filters)
        num_branches_pre = len(inputs)

        transition_outputs = []
        normalization = self.normalization.copy()
        normalization["trainable"] = trainable
      
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_filters_pre_layer[i] != filters[i]:
                    x = tf.keras.Sequential([
                        tf.keras.layers.Conv2D(filters[i], 3, 1, 
                                               data_format=self.data_format, 
                                               padding="same",
                                               use_bias=False,
                                               trainable=trainable,
                                               name="0"),
                        build_normalization(**normalization, name="1"),
                        build_activation(**self.activation, name="2")
                    ], name=name + "/%d" % i)(inputs[-1])
                else:
                    x = inputs[i]
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    in_filters = num_filters_pre_layer[-1]
                    out_filters = filters[i] if j == i - num_branches_pre else in_filters

                    conv3x3s.append(tf.keras.Sequential([
                        tf.keras.layers.Conv2D(out_filters, 3, 2, 
                                               padding="same",
                                               data_format=self.data_format, 
                                               use_bias=False,
                                               name="0"),
                        build_normalization(**normalization, name="1"),
                        build_activation(**self.activation, name="2")
                    ], name=str(j)))
                x = tf.keras.Sequential(conv3x3s, name=name + "/%d" % i)(inputs[-1])

            transition_outputs.append(x)

        return transition_outputs
    
    def _make_layer(self, inputs, block_fn, in_filters, filters, num_blocks, strides=1, dilation_rate=1, trainable=True, name="layers"):
        normalization = self.normalization.copy()
        normalization["trainable"] = trainable
        x = block_fn(filters, 
                     in_filters, 
                     strides=strides, 
                     dilation_rate=1,
                     data_format=self.data_format,
                     dropblock=self.dropblock,
                     normalization=normalization,
                     activation=self.activation,
                     trainable=trainable,
                     name=name + "/0")(inputs)
        in_filters = filters * block_fn.expansion
        for i in range(1, num_blocks):
            x = block_fn(filters, 
                         in_filters, 
                         strides=1, 
                         dilation_rate=dilation_rate,
                         data_format=self.data_format,
                         dropblock=self.dropblock,
                         normalization=normalization,
                         activation=self.activation,
                         trainable=trainable,
                         name=name + "/" + str(i))(x)
        return x
    
    def _make_stage(self, inputs, layer_config, in_filters, multi_scale_output=True, trainable=True, name="stage"):
        num_modules = layer_config["num_modules"]
        num_branches = layer_config["num_branches"]
        num_blocks = layer_config["num_blocks"]
        num_filters = layer_config["filters"]
        block_fn = BLOCKS[layer_config["block"]]
        fuse_method = layer_config["fuse_method"]

        normalization = self.normalization.copy()
        normalization["trainable"] = trainable
        x = inputs
        for i in range(num_modules):
            # multi_scale_output is onlye used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            x = HighResolutionModule(num_branches,
                                     block_fn,
                                     num_blocks,
                                     num_filters,
                                     in_filters,
                                     self.data_format,
                                     self.dropblock,
                                     normalization,
                                     self.activation,
                                     fuse_method,
                                     trainable,
                                     reset_multi_scale_output,
                                     name=name + "/%d" % i)(x)
        return x
    
    def _make_head(self, inputs, pre_stage_filters):
        head_filters = [32, 64, 128, 256]

        # Increasing the #filters on each resolution 
        # from C, 2C, 4C, 8C to 128, 256, 512, 1024
        x = self._make_layer(inputs[0], Bottleneck, pre_stage_filters[0], 
                             head_filters[0], 1, 1, name="incre_modules/0")
        for i in range(1, len(pre_stage_filters)):
            x1 = self._make_layer(inputs[i], Bottleneck, pre_stage_filters[i], 
                                  head_filters[i], 1, 1, name="incre_modules/%d" % i)
            out_filters = head_filters[i] * Bottleneck.expansion
            x2 = tf.keras.Sequential([
                        tf.keras.layers.Conv2D(out_filters, 3, 2, 
                                               data_format=self.data_format, 
                                               padding="same",
                                               name="0"),
                        build_normalization(**self.normalization, name="1"),
                        build_activation(**self.activation, name="2")], 
                        name="downsamp_modules/%d" % (i - 1))(x)
            x = tf.keras.layers.Add(name="sum%d" % i)([x1, x2])
        
        x = tf.keras.Sequential([
            tf.keras.layers.Conv2D(2048, 1, 1, 
                                   data_format=self.data_format, 
                                   name="0"),
            build_normalization(**self.normalization, name="1"),
            build_activation(**self.activation, name="2")], name="final_layer")(x)
        
        x = tf.keras.layers.GlobalAvgPool2D(data_format=self.data_format, name="gloabl_avgpool")(x)
        if self.drop_rate and self.drop_rate > 0.:
            x = tf.keras.layers.Dropout(rate=self.drop_rate, name="drop")(x)
        
        x = tf.keras.layers.Dense(units=self.num_classes, name="classifier")(x)

        return x
    
    def build_model(self):
        act = self.activation["activation"]
        norm = self.normalization["normalization"]
        trainable = 1 not in self.frozen_stages
        norm1 = self.normalization.copy()
        norm1["trainable"] = trainable

        x = tf.keras.layers.Lambda(
            lambda inp: (inp - self._rgb_mean) * (1. / self._rgb_std),
            name="norm_input")(self.img_input)
        x = tf.keras.layers.Conv2D(filters=64, 
                                   kernel_size=(3, 3), 
                                   strides=(2, 2), 
                                   padding="same", 
                                   data_format=self.data_format, 
                                   use_bias=False,
                                   name="conv1")(x)
        x = build_normalization(**norm1, name="%s1" % norm)(x)
        x = build_activation(**self.activation, name="%s1" % act)(x)
    
        x = tf.keras.layers.Conv2D(filters=64, 
                                   kernel_size=(3, 3), 
                                   strides=(2, 2), 
                                   padding="same", 
                                   data_format=self.data_format, 
                                   use_bias=False,
                                   name="conv2")(x)
        x = build_normalization(**norm1, name="%s2" % norm)(x)
        x = build_activation(**self.activation, name="%s2" % act)(x)

        x = self._make_layer(x, BLOCKS[self.cfg["stage1"]["block"]], 
                             64, self.cfg["stage1"]["filters"][0],
                             self.cfg["stage1"]["num_blocks"][0], 
                             trainable=2 not in self.frozen_stages, 
                             name="layer1")

        x_list = self._make_transition_layer([x], self.cfg["stage2"]["filters"], 
                                             3 not in self.frozen_stages, "transition1")
        x_list = self._make_stage(x_list, self.cfg["stage2"], self.cfg["stage2"]["filters"], 
                                  trainable=3 not in self.frozen_stages, name="stage2")

        x_list = self._make_transition_layer(x_list, self.cfg["stage3"]["filters"], 
                                             4 not in self.frozen_stages, "transition2")
        pre_stage_filters = [f * BLOCKS[self.cfg["stage3"]["block"]].expansion 
                             for f in self.cfg["stage3"]["filters"]]
        x_list = self._make_stage(x_list, self.cfg["stage3"], pre_stage_filters, 
                                  trainable=4 not in self.frozen_stages, name="stage3")
        
        x_list = self._make_transition_layer(x_list, self.cfg["stage4"]["filters"], 
                                             5 not in self.frozen_stages, "transition3")
        pre_stage_filters = [f * BLOCKS[self.cfg["stage4"]["block"]].expansion 
                             for f in self.cfg["stage4"]["filters"]]
        x_list = self._make_stage(x_list, self.cfg["stage4"], pre_stage_filters, 
                                  trainable=5 not in self.frozen_stages, name="stage4")
        
        outputs = self._make_head(x_list, pre_stage_filters)

        return tf.keras.Model(inputs=self.img_input, outputs=outputs, name=self.name)
    
    def init_weights(self, pretrained_weights_path):
        import os
        
        if os.path.exists(pretrained_weights_path):
            self.model.load_weights(pretrained_weights_path, by_name=True)
            tf.print("Initialized weights from", pretrained_weights_path)
        else:
            tf.print(pretrained_weights_path, "not exists! Initialized weights from scratch.")

    def torch2h5(self, torch_weights_path, output_path=None):
        import torch
        import numpy as np
        import torch.nn as nn

        net = torch.load(torch_weights_path)

        # for k, v in net.items():
        #     if "tracked" in k:
        #         continue
        #     print(k, v.shape) 
        
        for weight in self.model.weights:
            name = weight.name
            # print(name, weight.shape)
            name = name.split(":")[0]
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
            # print(name, weight.shape)

            tw = net[name].numpy()
            if len(tw.shape) == 4:
                tw = np.transpose(tw, (2, 3, 1, 0))
            
            if len(tw.shape) == 2:
                tw = np.transpose(tw, (1, 0))
            weight.assign(tw)

        del net

        self.model.save_weights(output_path)


@BACKBONES.register
class HRNetW18(HighResolutionNet):
    def __init__(self, 
                 convolution='conv2d', 
                 normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                 activation=dict(activation="relu"),
                 output_indices=(-1, ), 
                 strides=(2, 2, 2, 2, 2), 
                 dilation_rates=(1, 1, 1, 1, 1), 
                 frozen_stages=(-1, ), 
                 input_shape=None, 
                 input_tensor=None, 
                 dropblock=None, 
                 num_classes=1000, 
                 drop_rate=0.5):
        super(HRNetW18, self).__init__("hrnet_w18", 
                                       dict(stage1=dict(num_modules=1, num_branches=1, block='BOTTLENECK', 
                                                        num_blocks=(4, ), filters=(64, ), fuse_method='SUM'),
                                            stage2=dict(num_modules=1, num_branches=2, block='BASIC', 
                                                        num_blocks=(4, 4), filters=(18, 36), fuse_method='SUM'),
                                            stage3=dict(num_modules=4, num_branches=3, block='BASIC', 
                                                        num_blocks=(4, 4, 4), filters=(18, 36, 72), fuse_method='SUM'),
                                            stage4=dict(num_modules=3, num_branches=4, block='BASIC', 
                                                        num_blocks=(4, 4, 4, 4), filters=(18, 36, 72, 144), fuse_method='SUM')),
                                       convolution='conv2d', 
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
                                       drop_rate=drop_rate)


@BACKBONES.register
class HRNetW30(HighResolutionNet):
    def __init__(self, 
                 convolution='conv2d', 
                 normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                 activation=dict(activation="relu"),
                 output_indices=(-1, ), 
                 strides=(2, 2, 2, 2, 2), 
                 dilation_rates=(1, 1, 1, 1, 1), 
                 frozen_stages=(-1, ), 
                 input_shape=None, 
                 input_tensor=None, 
                 dropblock=None, 
                 num_classes=1000, 
                 drop_rate=0.5):
        super(HRNetW30, self).__init__("hrnet_w18", 
                                       dict(stage1=dict(num_modules=1, num_branches=1, block='BOTTLENECK', 
                                                        num_blocks=(4, ), filters=(64, ), fuse_method='SUM'),
                                            stage2=dict(num_modules=1, num_branches=2, block='BASIC', 
                                                        num_blocks=(4, 4), filters=(30, 60), fuse_method='SUM'),
                                            stage3=dict(num_modules=4, num_branches=3, block='BASIC', 
                                                        num_blocks=(4, 4, 4), filters=(30, 60, 120), fuse_method='SUM'),
                                            stage4=dict(num_modules=3, num_branches=4, block='BASIC', 
                                                        num_blocks=(4, 4, 4, 4), filters=(30, 60, 120, 240), fuse_method='SUM')),
                                       convolution='conv2d', 
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
                                       drop_rate=drop_rate)


@BACKBONES.register
class HRNetW32(HighResolutionNet):
    def __init__(self, 
                 convolution='conv2d', 
                 normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                 activation=dict(activation="relu"),
                 output_indices=(-1, ), 
                 strides=(2, 2, 2, 2, 2), 
                 dilation_rates=(1, 1, 1, 1, 1), 
                 frozen_stages=(-1, ), 
                 input_shape=None, 
                 input_tensor=None, 
                 dropblock=None, 
                 num_classes=1000, 
                 drop_rate=0.5):
        super(HRNetW32, self).__init__("hrnet_w18", 
                                       dict(stage1=dict(num_modules=1, num_branches=1, block='BOTTLENECK', 
                                                        num_blocks=(4, ), filters=(64, ), fuse_method='SUM'),
                                            stage2=dict(num_modules=1, num_branches=2, block='BASIC', 
                                                        num_blocks=(4, 4), filters=(32, 64), fuse_method='SUM'),
                                            stage3=dict(num_modules=4, num_branches=3, block='BASIC', 
                                                        num_blocks=(4, 4, 4), filters=(32, 64, 128), fuse_method='SUM'),
                                            stage4=dict(num_modules=3, num_branches=4, block='BASIC', 
                                                        num_blocks=(4, 4, 4, 4), filters=(32, 64, 128, 256), fuse_method='SUM')),
                                       convolution='conv2d', 
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
                                       drop_rate=drop_rate)


@BACKBONES.register
class HRNetW40(HighResolutionNet):
    def __init__(self, 
                 convolution='conv2d', 
                 normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                 activation=dict(activation="relu"),
                 output_indices=(-1, ), 
                 strides=(2, 2, 2, 2, 2), 
                 dilation_rates=(1, 1, 1, 1, 1), 
                 frozen_stages=(-1, ), 
                 input_shape=None, 
                 input_tensor=None, 
                 dropblock=None, 
                 num_classes=1000, 
                 drop_rate=0.5):
        super(HRNetW40, self).__init__("hrnet_w18", 
                                       dict(stage1=dict(num_modules=1, num_branches=1, block='BOTTLENECK', 
                                                        num_blocks=(4, ), filters=(64, ), fuse_method='SUM'),
                                            stage2=dict(num_modules=1, num_branches=2, block='BASIC', 
                                                        num_blocks=(4, 4), filters=(40, 80), fuse_method='SUM'),
                                            stage3=dict(num_modules=4, num_branches=3, block='BASIC', 
                                                        num_blocks=(4, 4, 4), filters=(40, 80, 160), fuse_method='SUM'),
                                            stage4=dict(num_modules=3, num_branches=4, block='BASIC', 
                                                        num_blocks=(4, 4, 4, 4), filters=(40, 80, 160, 320), fuse_method='SUM')),
                                       convolution='conv2d', 
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
                                       drop_rate=drop_rate)


@BACKBONES.register
class HRNetW44(HighResolutionNet):
    def __init__(self, 
                 convolution='conv2d', 
                 normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                 activation=dict(activation="relu"),
                 output_indices=(-1, ), 
                 strides=(2, 2, 2, 2, 2), 
                 dilation_rates=(1, 1, 1, 1, 1), 
                 frozen_stages=(-1, ), 
                 input_shape=None, 
                 input_tensor=None, 
                 dropblock=None, 
                 num_classes=1000, 
                 drop_rate=0.5):
        super(HRNetW44, self).__init__("hrnet_w18", 
                                       dict(stage1=dict(num_modules=1, num_branches=1, block='BOTTLENECK', 
                                                        num_blocks=(4, ), filters=(64, ), fuse_method='SUM'),
                                            stage2=dict(num_modules=1, num_branches=2, block='BASIC', 
                                                        num_blocks=(4, 4), filters=(44, 88), fuse_method='SUM'),
                                            stage3=dict(num_modules=4, num_branches=3, block='BASIC', 
                                                        num_blocks=(4, 4, 4), filters=(44, 88, 172), fuse_method='SUM'),
                                            stage4=dict(num_modules=3, num_branches=4, block='BASIC', 
                                                        num_blocks=(4, 4, 4, 4), filters=(44, 88, 172, 344), fuse_method='SUM')),
                                       convolution='conv2d', 
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
                                       drop_rate=drop_rate)


@BACKBONES.register
class HRNetW48(HighResolutionNet):
    def __init__(self, 
                 convolution='conv2d', 
                 normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                 activation=dict(activation="relu"),
                 output_indices=(-1, ), 
                 strides=(2, 2, 2, 2, 2), 
                 dilation_rates=(1, 1, 1, 1, 1), 
                 frozen_stages=(-1, ), 
                 input_shape=None, 
                 input_tensor=None, 
                 dropblock=None, 
                 num_classes=1000, 
                 drop_rate=0.5):
        super(HRNetW48, self).__init__("hrnet_w18", 
                                       dict(stage1=dict(num_modules=1, num_branches=1, block='BOTTLENECK', 
                                                        num_blocks=(4, ), filters=(64, ), fuse_method='SUM'),
                                            stage2=dict(num_modules=1, num_branches=2, block='BASIC', 
                                                        num_blocks=(4, 4), filters=(48, 96), fuse_method='SUM'),
                                            stage3=dict(num_modules=4, num_branches=3, block='BASIC', 
                                                        num_blocks=(4, 4, 4), filters=(48, 96, 192), fuse_method='SUM'),
                                            stage4=dict(num_modules=3, num_branches=4, block='BASIC', 
                                                        num_blocks=(4, 4, 4, 4), filters=(48, 96, 192, 384), fuse_method='SUM')),
                                       convolution='conv2d', 
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
                                       drop_rate=drop_rate)


@BACKBONES.register
class HRNetW64(HighResolutionNet):
    def __init__(self, 
                 convolution='conv2d', 
                 normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                 activation=dict(activation="relu"),
                 output_indices=(-1, ), 
                 strides=(2, 2, 2, 2, 2), 
                 dilation_rates=(1, 1, 1, 1, 1), 
                 frozen_stages=(-1, ), 
                 input_shape=None, 
                 input_tensor=None, 
                 dropblock=None, 
                 num_classes=1000, 
                 drop_rate=0.5):
        super(HRNetW64, self).__init__("hrnet_w18", 
                                       dict(stage1=dict(num_modules=1, num_branches=1, block='BOTTLENECK', 
                                                        num_blocks=(4, ), filters=(64, ), fuse_method='SUM'),
                                            stage2=dict(num_modules=1, num_branches=2, block='BASIC', 
                                                        num_blocks=(4, 4), filters=(64, 128), fuse_method='SUM'),
                                            stage3=dict(num_modules=4, num_branches=3, block='BASIC', 
                                                        num_blocks=(4, 4, 4), filters=(64, 128, 256), fuse_method='SUM'),
                                            stage4=dict(num_modules=3, num_branches=4, block='BASIC', 
                                                        num_blocks=(4, 4, 4, 4), filters=(64, 128, 256, 512), fuse_method='SUM')),
                                       convolution='conv2d', 
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
                                       drop_rate=drop_rate)


if __name__ == "__main__":
    hrnet = HRNetW18(input_shape=(224, 224, 3))

    hrnet.torch2h5("/home/bail/Downloads/hrnetv2_w18_imagenet_pretrained.pth",
                   "/home/bail/Workspace/pretrained_weights/hrnet_w18.h5")

    with tf.io.gfile.GFile("/home/bail/Documents/pandas.jpg", "rb") as gf:
        images = tf.image.decode_jpeg(gf.read())

    images = tf.image.convert_image_dtype(images, tf.float32)
    images = tf.image.resize(images, (224, 224))[None]

    cls = hrnet.model(images, training=False)
    cls = tf.nn.softmax(cls)
    print(tf.argmax(tf.squeeze(cls)))
    print(tf.reduce_max(cls))
    print(tf.nn.top_k(tf.squeeze(cls), k=5))
