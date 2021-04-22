import numpy as np
import tensorflow  as tf 
from .backbone import Backbone
from ..builder import BACKBONES
from core.layers import GroupConv2D
from core.layers import DropBlock2D
from core.layers import build_activation
from core.layers import build_normalization 


def get_stem_fn(stem_type):
    """Retrieves the stem function by name."""
    stem_fns = {
        "ResStemCifar": ResStemCifar,
        "ResStemImageNet": ResStemImageNet,
        "SimpleStemImageNet": SimpleResStemImageNet
    }

    err_str = "Stem type `{}` not supported."
    assert stem_type in stem_fns.keys(), err_str.format(stem_type)

    return stem_fns[stem_type]


def get_block_fn(block_type):
    block_fns = {
        "VanillaBlock": VanillaBlock,
        "ResBasicBlock": ResBasicBlock,
        "ResBottleneckBlock": ResBottleneckBlock
    }

    err_str = "Block type '{}' not supported"
    assert block_type in block_fns.keys(), err_str.format(block_type)

    return block_fns[block_type]


class AnyHead(tf.keras.Model):
    def __init__(self, num_classes, data_format="channels_last", **kwargs):
        super(AnyHead, self).__init__(**kwargs)

        self.avg_pool = tf.keras.layers.GlobalAvgPool2D(data_format=data_format, name="avg_pool")
        self.fc = tf.keras.layers.Dense(num_classes, name="fc")
    
    def call(self, inputs):
        x = self.avg_pool(inputs)
        x = self.fc(x)

        return x


class VanillaBlock(tf.keras.Model):
    """Vanilla block: [3x3 conv, BN, ReLU] x 2."""
    def __init__(self, 
                 filters, 
                 strides, 
                 dilation_rate=1,
                 data_format="channels_last",
                 dropblock=dict(block_size=7, drop_rate=0.1),
                 normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                 activation=dict(activation="relu"), 
                 trainable=True,
                 **kwargs):
        super(VanillaBlock, self).__init__(**kwargs)

        normalization = normalization.copy()
        normalization["trainable"] = trainable
        norm = normalization["normalization"]
        act = activation["activation"]

        self.a = tf.keras.layers.Conv2D(filters, 3, strides, "same", data_format, 
                                        dilation_rate, use_bias=False, trainable=trainable, name="a")
        self.a_bn = build_normalization(**normalization, name="a_%s" % norm)
        if dropblock is not None:
            self.a_dropblock = DropBlock2D(**dropblock, name="a_dropblock")
        self.a_relu = build_activation(**activation, name="a_%s" % act)
        self.b = tf.keras.layers.Conv2D(filters, 3, 1, "same", data_format, 
                                        dilation_rate, use_bias=False, trainable=trainable, name="b")
        self.b_bn = build_normalization(**normalization, name="b_%s" % norm)
        if dropblock is not None:
            self.b_dropblock = DropBlock2D(**dropblock, name="b_dropblock")

    def call(self, inputs, training=None):
        x = self.a(inputs)
        x = self.a_bn(x, training=training)
        if hasattr(self, "a_dropblock"):
            x = self.a_dropblock(x, training=training)
        x = self.a_relu(x)
        x = self.b(x)
        x = self.b_bn(x, training=training)
        if hasattr(self, "b_dropblock"):
            x = self.b_dropblock(x, training=training)
        
        return x


class BasicTransform(tf.keras.Model):
    """Basic transform: [3x3 conv, BN, ReLU] x 2."""
    def __init__(self, 
                 filters, 
                 strides, 
                 dilation_rate=1,
                 data_format="channels_last",
                 dropblock=dict(block_size=7, drop_rate=0.1),
                 normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                 activation=dict(activation="relu"), 
                 trainable=True,
                 **kwargs):
        super(BasicTransform, self).__init__(**kwargs)

        normalization = normalization.copy()
        normalization["trainable"] = trainable
        norm = normalization["normalization"]
        act = activation["activation"]

        self.a = tf.keras.layers.Conv2D(filters, 3, strides, "same", data_format, 
                                        dilation_rate, use_bias=False, trainable=trainable, name="a")
        self.a_bn = build_normalization(**normalization, name="a_%s" % norm)
        if dropblock is not None:
            self.a_dropblock = DropBlock2D(**dropblock, name="a_dropblock")
        self.a_relu = build_activation(**activation, name="a_%s" % act)
        self.b = tf.keras.layers.Conv2D(filters, 3, 1, "same", data_format, 
                                        dilation_rate, use_bias=False, trainable=trainable, name="b")
        self.b_bn = build_normalization(**normalization, name="b_%s" % norm)
        if dropblock is not None:
            self.b_dropblock = DropBlock2D(**dropblock, name="b_dropblock")

    def call(self, inputs, training=None):
        x = self.a(inputs)
        x = self.a_bn(x, training=training)
        if hasattr(self, "a_dropblock"):
            x = self.a_dropblock(x, training=training)
        x = self.a_relu(x)
        x = self.b(x)
        x = self.b_bn(x, training=training)
        if hasattr(self, "b_dropblock"):
            x = self.b_dropblock(x, training=training)
        
        return x


class ResBasicBlock(tf.keras.Model):
    """Residual basic block: x + F(x), F = basic transform."""
    def __init__(self, 
                 in_filters, 
                 filters, 
                 strides, 
                 dilation_rate=1,
                 data_format="channels_last",
                 dropblock=dict(block_size=7, drop_rate=0.1),
                 normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                 activation=dict(activation="relu"), 
                 trainable=True,
                 **kwargs):
        super(ResBasicBlock, self).__init__(**kwargs)

        normalization = normalization.copy()
        normalization["trainable"] = trainable
        norm = normalization["normalization"]
        act = activation["activation"]

        self.proj_block = (in_filters != filters) or (strides != 1)
        if self.proj_block:
            self.proj = tf.keras.layers.Conv2D(filters, 1, strides, "same", data_format, 
                                               dilation_rate, use_bias=False, trainable=trainable, name="proj")
            self.bn = build_normalization(**normalization, name="%s" % norm)
            if dropblock is not None:
                self.dropblock = DropBlock2D(**dropblock, name="dropblock")
        
        self.f = BasicTransform(filters, strides, dilation_rate, data_format, 
                                dropblock, normalization, activation, trainable, name="f")
        self.act = build_activation(**activation, name=act)
    
    def call(self, inputs, training=None):
        shortcut = inputs
        if self.proj_block:
            shortcut = self.bn(self.proj(shortcut), training=training)
            if hasattr(self, "dropblock"):
                shortcut = self.dropblock(shortcut, training=training)
        
        x = self.act(shortcut + self.f(x, training=training))

        return x


class SE(tf.keras.Model):
    """Squeeze-and-Excitation (SE) block: AvgPool, FC, ReLU, FC, Sigmoid."""
    def __init__(self, 
                 in_filters, 
                 se_filters, 
                 data_format="channels_last",
                 activation=dict(activation="relu"), 
                 trainable=True,
                 **kwargs):
        super(SE, self).__init__(**kwargs)

        se_axis = [1, 2] if data_format == "channels_last" else [2, 3]
        self.avg_pool = tf.keras.layers.Lambda(
            lambda inp: tf.reduce_mean(inp, se_axis, keepdims=True), name="avg_pool")
        self.se = tf.keras.Sequential([
            tf.keras.layers.Conv2D(se_filters, 1, trainable=trainable, name="0"),
            build_activation(**activation, name="1"),
            tf.keras.layers.Conv2D(in_filters, 1, trainable=trainable, name="2"),
            tf.keras.layers.Activation("sigmoid", name="3")
        ], name="f_ex")
    
    def call(self, inputs):
        x = self.avg_pool(inputs)
        x = self.se(x) * inputs

        return x


class BottleneckTransform(tf.keras.Model):
    """Bottleneck transformation: 1x1, 3x3 [+SE], 1x1."""

    def __init__(self,
                 in_filters, 
                 filters, 
                 strides, 
                 group_width,
                 se_ratio,
                 bottleneck_multiplier,
                 dilation_rate=1,
                 data_format="channels_last",
                 dropblock=dict(block_size=7, drop_rate=0.1),
                 normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                 activation=dict(activation="relu"), 
                 trainable=True,
                 **kwargs):
        super(BottleneckTransform, self).__init__(**kwargs)

        normalization = normalization.copy()
        normalization["trainable"] = trainable
        norm = normalization["normalization"]
        act = activation["activation"]

        bottleneck_width = int(round(filters * bottleneck_multiplier))
        group = bottleneck_width // group_width
        self.a = tf.keras.layers.Conv2D(bottleneck_width, 1, 1, "same", data_format, dilation_rate, 
                                        use_bias=False, trainable=trainable, name="a")
        self.a_bn = build_normalization(**normalization, name="a_%s" % norm)
        if dropblock is not None:
            self.a_dropblock = DropBlock2D(**dropblock, name="a_dropblock")
        self.a_relu = build_activation(**activation, name="a_%s" % act)
        
        self.b = tf.keras.layers.Conv2D(bottleneck_width, 3, strides, "same", data_format, dilation_rate, 
                                        groups=group, use_bias=False, trainable=trainable, name="b")
        self.b_bn = build_normalization(**normalization, name="b_%s" % norm)
        if dropblock is not None:
            self.b_dropblock = DropBlock2D(**dropblock, name="b_dropblock")
        self.b_relu = build_activation(**activation, name="b_%s" % act)

        if se_ratio:
            se_width = int(round(in_filters * se_ratio))
            self.se = SE(bottleneck_width, se_width, data_format, activation, trainable, name="se")

        self.c = tf.keras.layers.Conv2D(filters, 1, 1, "same", data_format, 
                                        dilation_rate, use_bias=False, trainable=trainable, name="c")
        self.c_bn = build_normalization(**normalization, name="c_%s" % norm)
        if dropblock is not None:
            self.c_dropblock = DropBlock2D(**dropblock, name="c_dropblock")
    
    def call(self, inputs, training=None):
        x = self.a(inputs)
        x = self.a_bn(x, training=training)
        if hasattr(self, "a_dropblock"):
            x = self.a_dropblock(x, training=training)
        x = self.a_relu(x)

        x = self.b(x)
        x = self.b_bn(x, training=training)
        if hasattr(self, "b_dropblock"):
            x = self.b_dropblock(x, training=training)
        x = self.b_relu(x)

        if hasattr(self, "se"):
            x = self.se(x)
        
        x = self.c(x)
        x = self.c_bn(x, training=training)
        if hasattr(self, "c_dropblock"):
            x = self.c_dropblock(x, training=training)

        return x


class ResBottleneckBlock(tf.keras.Model):
    """Residual bottleneck block: x + F(x), F = bottleneck transform."""
    def __init__(self, 
                 in_filters, 
                 filters, 
                 group_width=1,
                 se_ratio=None,
                 bottleneck_multiplier=1.0,
                 strides=1, 
                 dilation_rate=1,
                 data_format="channels_last",
                 dropblock=dict(block_size=7, drop_rate=0.1),
                 normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                 activation=dict(activation="relu"), 
                 trainable=True,
                 **kwargs):
        super(ResBottleneckBlock, self).__init__(**kwargs)
        
        normalization = normalization.copy()
        normalization["trainable"] = trainable
        norm = normalization["normalization"]
        act = activation["activation"]
        
        self.proj_block = (in_filters != filters) or (strides != 1)
        if self.proj_block:
            self.proj = tf.keras.layers.Conv2D(filters, 1, strides, "same", data_format, 
                                               dilation_rate, use_bias=False, trainable=trainable, name="proj")
            self.bn = build_normalization(**normalization, name="%s" % norm)
            if dropblock is not None:
                self.dropblock = DropBlock2D(**dropblock, name="dropblock")

        self.f = BottleneckTransform(in_filters, filters, strides, group_width, se_ratio,
                                     bottleneck_multiplier, dilation_rate, data_format,
                                     dropblock, normalization, activation, trainable, name="f")
        self.act = build_activation(**activation, name=act)
    
    def call(self, inputs, training=None):
        shortcut = inputs
        if self.proj_block:
            shortcut = self.bn(self.proj(shortcut), training=training)
            if hasattr(self, "dropblock"):
                shortcut = self.dropblock(shortcut, training=training)
        
        x = self.act(shortcut + self.f(inputs, training=training))

        return x


class ResStemCifar(tf.keras.Model):
    def __init__(self, 
                 filters, 
                 data_format="channels_last",
                 dropblock=dict(block_size=7, drop_rate=0.1),
                 normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                 activation=dict(activation="relu"), 
                 trainable=True,
                 **kwargs):
        super(ResStemCifar, self).__init__(**kwargs)

        normalization = normalization.copy()
        normalization["trainable"] = trainable
        norm = normalization["normalization"]
        act = activation["activation"]

        self.conv = tf.keras.layers.Conv2D(filters=filters, 
                                           kernel_size=3, 
                                           strides=1, 
                                           padding="same", 
                                           data_format=data_format, 
                                           use_bias=False, 
                                           trainable=trainable, 
                                           name="conv")
        self.bn = build_normalization(**normalization, name=norm)
        if dropblock is not None:
            self.dropblock = DropBlock2D(**dropblock, name="dropblock")
        self.relu = build_activation(**activation, name=act)
    
    def call(self, inputs, training=None):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        if hasattr(self, "dropblock"):
            x = self.dropblock(x, training=training)
        x = self.relu(x)

        return x


class ResStemImageNet(tf.keras.Model):
    def __init__(self, 
                 filters, 
                 data_format="channels_last",
                 dropblock=dict(block_size=7, drop_rate=0.1),
                 normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                 activation=dict(activation="relu"), 
                 trainable=True,
                 **kwargs):
        super(ResStemImageNet, self).__init__(**kwargs)

        normalization = normalization.copy()
        normalization["trainable"] = trainable
        norm = normalization["normalization"]
        act = activation["activation"]

        self.conv = tf.keras.layers.Conv2D(ilters=filters, 
                                           kernel_size=7, 
                                           strides=2, 
                                           padding="same", 
                                           data_format=data_format, 
                                           use_bias=False, 
                                           trainable=trainable, 
                                           name="conv")
        self.bn = build_normalization(**normalization, name=norm)
        if dropblock is not None:
            self.dropblock = DropBlock2D(**dropblock, name="dropblock")
        self.relu = build_activation(**activation, name=act)
        self.pool = tf.keras.layers.MaxPool2D(3, 2, "same", name="pool")
    
    def call(self, inputs, training=None):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        if hasattr(self, "dropblock"):
            x = self.dropblock(x, training=training)
        x = self.relu(x)
        x = self.pool(x)

        return x


class SimpleResStemImageNet(tf.keras.Model):
    def __init__(self, 
                 filters, 
                 data_format="channels_last",
                 dropblock=dict(block_size=7, drop_rate=0.1),
                 normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                 activation=dict(activation="relu"), 
                 trainable=True,
                 **kwargs):
        super(SimpleResStemImageNet, self).__init__(**kwargs)

        normalization = normalization.copy()
        normalization["trainable"] = trainable
        norm = normalization["normalization"]
        act = activation["activation"]

        self.conv = tf.keras.layers.Conv2D(filters=filters, 
                                           kernel_size=3, 
                                           strides=2, 
                                           padding="same", 
                                           data_format=data_format, 
                                           use_bias=False, 
                                           trainable=trainable, 
                                           name="conv")
        self.bn = build_normalization(**normalization, name=norm)
        if dropblock is not None:
            self.dropblock = DropBlock2D(**dropblock, name="dropblock")
        self.relu = build_activation(**activation, name=act)
    
    def call(self, inputs, training=None):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        if hasattr(self, "dropblock"):
            x = self.dropblock(x, training=training)
        x = self.relu(x)

        return x


class AnyStage(tf.keras.Model):
    """AnyNet stage (sequence of blocks w/ the same output shape)."""

    def __init__(self,
                 in_filters, 
                 filters, 
                 depth,
                 block_fn,
                 group_width=1,
                 se_ratio=None,
                 bottleneck_multiplier=1.0,
                 strides=1, 
                 dilation_rate=1,
                 data_format="channels_last",
                 dropblock=dict(block_size=7, drop_rate=0.1),
                 normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                 activation=dict(activation="relu"), 
                 trainable=True,
                 **kwargs):
        super(AnyStage, self).__init__(**kwargs)
        for i in range(depth):
            b_strides = strides if i == 0 else 1
            b_in_filters = in_filters if i == 0 else filters

            name = "b{}".format(i+1)
            
            setattr(self, name, block_fn(in_filters=b_in_filters,
                                         filters=filters,
                                         group_width=group_width,
                                         se_ratio=se_ratio,
                                         bottleneck_multiplier=bottleneck_multiplier,
                                         strides=b_strides, 
                                         dilation_rate=dilation_rate,
                                         data_format=data_format,
                                         dropblock=dropblock,
                                         normalization=normalization,
                                         activation=activation, 
                                         trainable=trainable,
                                         name=name))
        self.depth = depth
    
    def call(self, inputs, training=None):
        x = inputs
        for i in range(self.depth):
            x = getattr(self, "b%d" % (i + 1))(x, training=training)
        
        return x


class AnyNet(Backbone):
    """RegNet
    
    Args:
        w_0 (int): Initial width
        w_a (float): Slope
        w_m (float): Quantization        
        group_w (int): Group width
        bot_mul (int): Bottleneck multiplier (bm = 1 / b from the paper)
    """
    def __init__(self, 
                 name,
                 w_0=32,
                 w_a=5.0,
                 w_m=2.5,
                 group_w=8,
                 use_se=False,
                 se_ratio=0.25,
                 bot_mul=1.0,
                 depth=13,
                 stem_type="SimpleStemImageNet", 
                 stem_filters=32, 
                 block_type="ResBottleneckBlock",
                 convolution='conv2d', 
                 normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                 activation=dict(activation="relu"),
                 output_indices=(3,4), 
                 strides=(2, 2, 2, 2, 2), 
                 dilation_rates=(1, 1, 1, 1, 1), 
                 frozen_stages=(-1, ), 
                 input_shape=None, 
                 input_tensor=None, 
                 dropblock=None, 
                 num_classes=1000, 
                 drop_rate=0.5):
        super(AnyNet, self).__init__(name, 
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
        
        # Generate RegNet ws per block
        widths, num_stages, _, _ = self.generate_regnet(w_a, w_0, w_m, depth)
        # Convert to per stage format
        stage_widths, stage_depths = self.get_stages_from_blocks(widths, widths)
        stage_group_widths = [group_w for _ in range(num_stages)]
        stage_bot_mul = [bot_mul for _ in range(num_stages)]
        stage_strides = [2 for _ in range(num_stages)]
        # Adjust the compatibility of ws and gws
        stage_widths, stage_group_widths = self.adjust_ws_gs_comp(stage_widths, stage_bot_mul, stage_group_widths)

        self.stem_type = stem_type
        self.stem_filters = stem_filters
        self.block_fn = block_type
        self.strides = stage_strides
        self.widths = stage_widths
        self.depths = stage_depths
        self.bottleneck_multipliers = stage_bot_mul
        self.group_widths = stage_group_widths
        self.se_ratio = se_ratio if use_se else None

        self.stem_fn = get_stem_fn(stem_type)
        self.block_fn = get_block_fn(block_type)

    def build_model(self):
        act = self.activation["activation"]
        norm = self.normalization["normalization"]
        trainable = 1 not in self.frozen_stages
        norm1 = self.normalization.copy()
        norm1["trainable"] = trainable
        
        def _norm(inp):
            inp -= tf.constant([0.485, 0.456, 0.406], tf.float32, [1, 1, 1, 3])
            inp /= tf.constant([0.229, 0.224, 0.225], tf.float32, [1, 1, 1, 3])

            return inp
        
        x = tf.keras.layers.Lambda(_norm, name="norm_input")(self.img_input)
        
        trainable = 1 not in self.frozen_stages
        norm1 = self.normalization.copy()
        norm1["trainable"] = trainable
        x = self.stem_fn(filters=self.stem_filters,
                         data_format=self.data_format,
                         dropblock=self.dropblock,
                         normalization=norm1,
                         activation=self.activation, 
                         trainable=trainable,
                         name="stem")(x)
        
        in_filters = self.stem_filters
        stage_params = zip(self.depths, self.widths, self.strides, self.dilation_rates,
                           self.bottleneck_multipliers, self.group_widths)
        for i, (d, w, s, dr, bm, gw) in enumerate(stage_params):
            # print(i, in_filters, d, w, s, bm, gw)
            x = AnyStage(in_filters, w, d, self.block_fn, gw, self.se_ratio, bm, s, 1,
                         self.data_format, self.dropblock, self.normalization, self.activation, 
                         (i+2) not in self.frozen_stages, name="s%d" % (i+1))(x)
            in_filters = w
        
        x = AnyHead(self.num_classes, self.data_format, name="head")(x)

        return tf.keras.Model(inputs=self.img_input, outputs=x, name=self.name)

    def quantize_float(self, f, q):
        """Converts a float to closest non-zero int divisible by q."""
        return int(round(f / q) * q)
    
    def adjust_ws_gs_comp(self, ws, bms, gs):
        """Adjusts the compatibility of widths and groups."""
        ws_bot = [int(w * b) for w, b in zip(ws, bms)]
        gs = [min(g, w_bot) for g, w_bot in zip(gs, ws_bot)]
        ws_bot = [self.quantize_float(w_bot, g) for w_bot, g in zip(ws_bot, gs)]
        ws = [int(w_bot / b) for w_bot, b in zip(ws_bot, bms)]

        return ws, gs
    
    def get_stages_from_blocks(self, ws, rs):
        """"Gets ws/ds of network at each stage from per block values."""
        ts_temp = zip(ws + [0], [0] + ws, rs + [0], [0] + rs)
        ts = [w != wp or r != rp for w, wp, r, rp in ts_temp]
        s_ws = [w for w, t in zip(ws, ts[:-1]) if t]
        s_ds = np.diff([d for d, t in zip(range(len(ts)), ts) if t]).tolist()
        
        return s_ws, s_ds
    
    def generate_regnet(self, w_a, w_0, w_m, d, q=8):
        """Generates per block ws from RegNet parameters."""
        assert w_a >= 0 and w_0 > 0 and w_m > 1 and w_0 % q == 0

        ws_cont = np.arange(d) * w_a + w_0
        ks = np.round(np.log(ws_cont / w_0) / np.log(w_m))
        ws = w_0 * np.power(w_m, ks)
        ws = np.round(np.divide(ws, q)) * q
        num_stages, max_stage = len(np.unique(ws)), ks.max() + 1
        ws, ws_cont = ws.astype(int).tolist(), ws_cont.tolist()

        return ws, num_stages, max_stage, ws_cont

    def torch2h5(self, torch_weights_path, output_path=None):
        import torch
        import numpy as np
        import torch.nn as nn

        net = torch.load(torch_weights_path)

        # for k, v in net["model_state"].items():
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

            tw = net["model_state"][name].numpy()
            if len(tw.shape) == 4:
                tw = np.transpose(tw, (2, 3, 1, 0))
            
            if len(tw.shape) == 2:
                tw = np.transpose(tw, (1, 0))
            weight.assign(tw)

        del net

        self.model.save_weights(output_path)


@BACKBONES.register
class RegNetX200MF(AnyNet):
    def __init__(self, 
                 convolution="conv2d",
                 normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                 activation=dict(activation="relu"),
                 output_indices=(3, 4),
                 strides=(2, 2, 2, 2, 2),
                 dilation_rates=(1, 1, 1, 1, 1),
                 frozen_stages=(-1,),
                 input_shape=(224, 224, 3),
                 input_tensor=None,
                 dropblock=None,
                 num_classes=1000,
                 drop_rate=0.5):
        super(RegNetX200MF, self).__init__("RegNetX200MF",
                                           w_0=24,
                                           w_a=36.44,
                                           w_m=2.49,
                                           group_w=8,
                                           use_se=False,
                                           se_ratio=0.25,
                                           bot_mul=1.0,
                                           depth=13,
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


@BACKBONES.register
class RegNetX400MF(AnyNet):
    def __init__(self, 
                 convolution="conv2d",
                 normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                 activation=dict(activation="relu"),
                 output_indices=(3, 4),
                 strides=(2, 2, 2, 2, 2),
                 dilation_rates=(1, 1, 1, 1, 1),
                 frozen_stages=(-1,),
                 input_shape=(224, 224, 3),
                 input_tensor=None,
                 dropblock=None,
                 num_classes=1000,
                 drop_rate=0.5):
        super(RegNetX400MF, self).__init__("RegNetX400MF",
                                           w_0=24,
                                           w_a=24.48,
                                           w_m=2.54,
                                           group_w=16,
                                           use_se=False,
                                           se_ratio=0.25,
                                           bot_mul=1.0,
                                           depth=22,
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


@BACKBONES.register
class RegNetX600MF(AnyNet):
    def __init__(self, 
                 convolution="conv2d",
                 normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                 activation=dict(activation="relu"),
                 output_indices=(3, 4),
                 strides=(2, 2, 2, 2, 2),
                 dilation_rates=(1, 1, 1, 1, 1),
                 frozen_stages=(-1,),
                 input_shape=(224, 224, 3),
                 input_tensor=None,
                 dropblock=None,
                 num_classes=1000,
                 drop_rate=0.5):
        super(RegNetX600MF, self).__init__("RegNetX600MF",
                                           w_0=48,
                                           w_a=36.97,
                                           w_m=2.24,
                                           group_w=24,
                                           use_se=False,
                                           se_ratio=0.25,
                                           bot_mul=1.0,
                                           depth=16,
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


@BACKBONES.register
class RegNetX800MF(AnyNet):
    def __init__(self, 
                 convolution="conv2d",
                 normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                 activation=dict(activation="relu"),
                 output_indices=(3, 4),
                 strides=(2, 2, 2, 2, 2),
                 dilation_rates=(1, 1, 1, 1, 1),
                 frozen_stages=(-1,),
                 input_shape=(224, 224, 3),
                 input_tensor=None,
                 dropblock=None,
                 num_classes=1000,
                 drop_rate=0.5):
        super(RegNetX800MF, self).__init__("RegNetX800MF",
                                           w_0=56,
                                           w_a=35.73,
                                           w_m=2.28,
                                           group_w=16,
                                           use_se=False,
                                           se_ratio=0.25,
                                           bot_mul=1.0,
                                           depth=16,
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


@BACKBONES.register
class RegNetX1Point6GF(AnyNet):
    def __init__(self, 
                 convolution="conv2d",
                 normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                 activation=dict(activation="relu"),
                 output_indices=(3, 4),
                 strides=(2, 2, 2, 2, 2),
                 dilation_rates=(1, 1, 1, 1, 1),
                 frozen_stages=(-1,),
                 input_shape=(224, 224, 3),
                 input_tensor=None,
                 dropblock=None,
                 num_classes=1000,
                 drop_rate=0.5):
        super(RegNetX1Point6GF, self).__init__("RegNetX1.6GF",
                                               w_0=80,
                                               w_a=34.01,
                                               w_m=2.25,
                                               group_w=24,
                                               use_se=False,
                                               se_ratio=0.25,
                                               bot_mul=1.0,
                                               depth=18,
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


@BACKBONES.register
class RegNetX3Point2GF(AnyNet):
    def __init__(self, 
                 convolution="conv2d",
                 normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                 activation=dict(activation="relu"),
                 output_indices=(3, 4),
                 strides=(2, 2, 2, 2, 2),
                 dilation_rates=(1, 1, 1, 1, 1),
                 frozen_stages=(-1,),
                 input_shape=(224, 224, 3),
                 input_tensor=None,
                 dropblock=None,
                 num_classes=1000,
                 drop_rate=0.5):
        super(RegNetX3Point2GF, self).__init__("RegNetX3.2GF",
                                               w_0=88,
                                               w_a=26.31,
                                               w_m=2.25,
                                               group_w=48,
                                               use_se=False,
                                               se_ratio=0.25,
                                               bot_mul=1.0,
                                               depth=25,
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


@BACKBONES.register
class RegNetX4Point0GF(AnyNet):
    def __init__(self, 
                 convolution="conv2d",
                 normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                 activation=dict(activation="relu"),
                 output_indices=(3, 4),
                 strides=(2, 2, 2, 2, 2),
                 dilation_rates=(1, 1, 1, 1, 1),
                 frozen_stages=(-1,),
                 input_shape=(224, 224, 3),
                 input_tensor=None,
                 dropblock=None,
                 num_classes=1000,
                 drop_rate=0.5):
        super(RegNetX4Point0GF, self).__init__("RegNetX4.0GF",
                                               w_0=96,
                                               w_a=38.65,
                                               w_m=2.43,
                                               group_w=40,
                                               use_se=False,
                                               se_ratio=0.25,
                                               bot_mul=1.0,
                                               depth=23,
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


@BACKBONES.register
class RegNetX6Point4GF(AnyNet):
    def __init__(self, 
                 convolution="conv2d",
                 normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                 activation=dict(activation="relu"),
                 output_indices=(3, 4),
                 strides=(2, 2, 2, 2, 2),
                 dilation_rates=(1, 1, 1, 1, 1),
                 frozen_stages=(-1,),
                 input_shape=(224, 224, 3),
                 input_tensor=None,
                 dropblock=None,
                 num_classes=1000,
                 drop_rate=0.5):
        super(RegNetX6Point4GF, self).__init__("RegNetX6.4GF",
                                               w_0=184,
                                               w_a=60.83,
                                               w_m=2.07,
                                               group_w=56,
                                               use_se=False,
                                               se_ratio=0.25,
                                               bot_mul=1.0,
                                               depth=17,
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


@BACKBONES.register
class RegNetX8Point0GF(AnyNet):
    def __init__(self, 
                 convolution="conv2d",
                 normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                 activation=dict(activation="relu"),
                 output_indices=(3, 4),
                 strides=(2, 2, 2, 2, 2),
                 dilation_rates=(1, 1, 1, 1, 1),
                 frozen_stages=(-1,),
                 input_shape=(224, 224, 3),
                 input_tensor=None,
                 dropblock=None,
                 num_classes=1000,
                 drop_rate=0.5):
        super(RegNetX8Point0GF, self).__init__("RegNetX8.0GF",
                                               w_0=80,
                                               w_a=49.56,
                                               w_m=2.88,
                                               group_w=120,
                                               use_se=False,
                                               se_ratio=0.25,
                                               bot_mul=1.0,
                                               depth=23,
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


@BACKBONES.register
class RegNetX12GF(AnyNet):
    def __init__(self, 
                 convolution="conv2d",
                 normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                 activation=dict(activation="relu"),
                 output_indices=(3, 4),
                 strides=(2, 2, 2, 2, 2),
                 dilation_rates=(1, 1, 1, 1, 1),
                 frozen_stages=(-1,),
                 input_shape=(224, 224, 3),
                 input_tensor=None,
                 dropblock=None,
                 num_classes=1000,
                 drop_rate=0.5):
        super(RegNetX12GF, self).__init__("RegNetX12GF",
                                          w_0=168,
                                          w_a=73.36,
                                          w_m=2.37,
                                          group_w=112,
                                          use_se=False,
                                          se_ratio=0.25,
                                          bot_mul=1.0,
                                          depth=19,
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


@BACKBONES.register
class RegNetX16GF(AnyNet):
    def __init__(self, 
                 convolution="conv2d",
                 normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                 activation=dict(activation="relu"),
                 output_indices=(3, 4),
                 strides=(2, 2, 2, 2, 2),
                 dilation_rates=(1, 1, 1, 1, 1),
                 frozen_stages=(-1,),
                 input_shape=(224, 224, 3),
                 input_tensor=None,
                 dropblock=None,
                 num_classes=1000,
                 drop_rate=0.5):
        super(RegNetX16GF, self).__init__("RegNetX16GF",
                                          w_0=216,
                                          w_a=55.59,
                                          w_m=2.1,
                                          group_w=128,
                                          use_se=False,
                                          se_ratio=0.25,
                                          bot_mul=1.0,
                                          depth=22,
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


@BACKBONES.register
class RegNetX32GF(AnyNet):
    def __init__(self, 
                 convolution="conv2d",
                 normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                 activation=dict(activation="relu"),
                 output_indices=(3, 4),
                 strides=(2, 2, 2, 2, 2),
                 dilation_rates=(1, 1, 1, 1, 1),
                 frozen_stages=(-1,),
                 input_shape=(224, 224, 3),
                 input_tensor=None,
                 dropblock=None,
                 num_classes=1000,
                 drop_rate=0.5):
        super(RegNetX32GF, self).__init__("RegNetX32GF",
                                          w_0=320,
                                          w_a=69.86,
                                          w_m=2.0,
                                          group_w=168,
                                          use_se=False,
                                          se_ratio=0.25,
                                          bot_mul=1.0,
                                          depth=23,
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


@BACKBONES.register
class RegNetY200MF(AnyNet):
    def __init__(self, 
                 convolution="conv2d",
                 normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                 activation=dict(activation="relu"),
                 output_indices=(3, 4),
                 strides=(2, 2, 2, 2, 2),
                 dilation_rates=(1, 1, 1, 1, 1),
                 frozen_stages=(-1,),
                 input_shape=(224, 224, 3),
                 input_tensor=None,
                 dropblock=None,
                 num_classes=1000,
                 drop_rate=0.5):
        super(RegNetY200MF, self).__init__("RegNetY200MF",
                                           w_0=24,
                                           w_a=36.44,
                                           w_m=2.49,
                                           group_w=8,
                                           use_se=True,
                                           se_ratio=0.25,
                                           bot_mul=1.0,
                                           depth=13,
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


@BACKBONES.register
class RegNetY400MF(AnyNet):
    def __init__(self, 
                 convolution="conv2d",
                 normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                 activation=dict(activation="relu"),
                 output_indices=(3, 4),
                 strides=(2, 2, 2, 2, 2),
                 dilation_rates=(1, 1, 1, 1, 1),
                 frozen_stages=(-1,),
                 input_shape=(224, 224, 3),
                 input_tensor=None,
                 dropblock=None,
                 num_classes=1000,
                 drop_rate=0.5):
        super(RegNetY400MF, self).__init__("RegNetY400MF",
                                           w_0=48,
                                           w_a=27.89,
                                           w_m=2.09,
                                           group_w=8,
                                           use_se=True,
                                           se_ratio=0.25,
                                           bot_mul=1.0,
                                           depth=16,
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


@BACKBONES.register
class RegNetY600MF(AnyNet):
    def __init__(self, 
                 convolution="conv2d",
                 normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                 activation=dict(activation="relu"),
                 output_indices=(3, 4),
                 strides=(2, 2, 2, 2, 2),
                 dilation_rates=(1, 1, 1, 1, 1),
                 frozen_stages=(-1,),
                 input_shape=(224, 224, 3),
                 input_tensor=None,
                 dropblock=None,
                 num_classes=1000,
                 drop_rate=0.5):
        super(RegNetY600MF, self).__init__("RegNetY600MF",
                                           w_0=48,
                                           w_a=32.54,
                                           w_m=2.32,
                                           group_w=16,
                                           use_se=True,
                                           se_ratio=0.25,
                                           bot_mul=1.0,
                                           depth=15,
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


@BACKBONES.register
class RegNetY800MF(AnyNet):
    def __init__(self, 
                 convolution="conv2d",
                 normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                 activation=dict(activation="relu"),
                 output_indices=(3, 4),
                 strides=(2, 2, 2, 2, 2),
                 dilation_rates=(1, 1, 1, 1, 1),
                 frozen_stages=(-1,),
                 input_shape=(224, 224, 3),
                 input_tensor=None,
                 dropblock=None,
                 num_classes=1000,
                 drop_rate=0.5):
        super(RegNetY800MF, self).__init__("RegNetY800MF",
                                           w_0=56,
                                           w_a=38.84,
                                           w_m=2.4,
                                           group_w=16,
                                           use_se=True,
                                           se_ratio=0.25,
                                           bot_mul=1.0,
                                           depth=14,
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


@BACKBONES.register
class RegNetY1Point6GF(AnyNet):
    def __init__(self, 
                 convolution="conv2d",
                 normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                 activation=dict(activation="relu"),
                 output_indices=(3, 4),
                 strides=(2, 2, 2, 2, 2),
                 dilation_rates=(1, 1, 1, 1, 1),
                 frozen_stages=(-1,),
                 input_shape=(224, 224, 3),
                 input_tensor=None,
                 dropblock=None,
                 num_classes=1000,
                 drop_rate=0.5):
        super(RegNetY1Point6GF, self).__init__("RegNetY1.6GF",
                                               w_0=48,
                                               w_a=20.71,
                                               w_m=2.65,
                                               group_w=24,
                                               use_se=True,
                                               se_ratio=0.25,
                                               bot_mul=1.0,
                                               depth=27,
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


@BACKBONES.register
class RegNetY3Point2GF(AnyNet):
    def __init__(self, 
                 convolution="conv2d",
                 normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                 activation=dict(activation="relu"),
                 output_indices=(3, 4),
                 strides=(2, 2, 2, 2, 2),
                 dilation_rates=(1, 1, 1, 1, 1),
                 frozen_stages=(-1,),
                 input_shape=(224, 224, 3),
                 input_tensor=None,
                 dropblock=None,
                 num_classes=1000,
                 drop_rate=0.5):
        super(RegNetY3Point2GF, self).__init__("RegNetY3.2GF",
                                               w_0=80,
                                               w_a=42.63,
                                               w_m=2.66,
                                               group_w=24,
                                               use_se=True,
                                               se_ratio=0.25,
                                               bot_mul=1.0,
                                               depth=21,
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


@BACKBONES.register
class RegNetY4Point0GF(AnyNet):
    def __init__(self, 
                 convolution="conv2d",
                 normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                 activation=dict(activation="relu"),
                 output_indices=(3, 4),
                 strides=(2, 2, 2, 2, 2),
                 dilation_rates=(1, 1, 1, 1, 1),
                 frozen_stages=(-1,),
                 input_shape=(224, 224, 3),
                 input_tensor=None,
                 dropblock=None,
                 num_classes=1000,
                 drop_rate=0.5):
        super(RegNetY6Point4GF, self).__init__("RegNetY4.0GF",
                                               w_0=96,
                                               w_a=31.41,
                                               w_m=2.24,
                                               group_w=64,
                                               use_se=True,
                                               se_ratio=0.25,
                                               bot_mul=1.0,
                                               depth=22,
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


@BACKBONES.register
class RegNetY6Point4GF(AnyNet):
    def __init__(self, 
                 convolution="conv2d",
                 normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                 activation=dict(activation="relu"),
                 output_indices=(3, 4),
                 strides=(2, 2, 2, 2, 2),
                 dilation_rates=(1, 1, 1, 1, 1),
                 frozen_stages=(-1,),
                 input_shape=(224, 224, 3),
                 input_tensor=None,
                 dropblock=None,
                 num_classes=1000,
                 drop_rate=0.5):
        super(RegNetY6Point4GF, self).__init__("RegNetY6.4GF",
                                               w_0=112,
                                               w_a=33.22,
                                               w_m=2.27,
                                               group_w=72,
                                               use_se=True,
                                               se_ratio=0.25,
                                               bot_mul=1.0,
                                               depth=25,
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


@BACKBONES.register
class RegNetY8Point0GF(AnyNet):
    def __init__(self, 
                 convolution="conv2d",
                 normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                 activation=dict(activation="relu"),
                 output_indices=(3, 4),
                 strides=(2, 2, 2, 2, 2),
                 dilation_rates=(1, 1, 1, 1, 1),
                 frozen_stages=(-1,),
                 input_shape=(224, 224, 3),
                 input_tensor=None,
                 dropblock=None,
                 num_classes=1000,
                 drop_rate=0.5):
        super(RegNetY8Point0GF, self).__init__("RegNetY8.0GF",
                                               w_0=192,
                                               w_a=76.82,
                                               w_m=2.19,
                                               group_w=56,
                                               use_se=True,
                                               se_ratio=0.25,
                                               bot_mul=1.0,
                                               depth=17,
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


@BACKBONES.register
class RegNetY12GF(AnyNet):
    def __init__(self, 
                 convolution="conv2d",
                 normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                 activation=dict(activation="relu"),
                 output_indices=(3, 4),
                 strides=(2, 2, 2, 2, 2),
                 dilation_rates=(1, 1, 1, 1, 1),
                 frozen_stages=(-1,),
                 input_shape=(224, 224, 3),
                 input_tensor=None,
                 dropblock=None,
                 num_classes=1000,
                 drop_rate=0.5):
        super(RegNetY12GF, self).__init__("RegNetY12GF",
                                          w_0=168,
                                          w_a=73.36,
                                          w_m=2.37,
                                          group_w=112,
                                          use_se=True,
                                          se_ratio=0.25,
                                          bot_mul=1.0,
                                          depth=19,
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


@BACKBONES.register
class RegNetY16GF(AnyNet):
    def __init__(self, 
                 convolution="conv2d",
                 normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                 activation=dict(activation="relu"),
                 output_indices=(3, 4),
                 strides=(2, 2, 2, 2, 2),
                 dilation_rates=(1, 1, 1, 1, 1),
                 frozen_stages=(-1,),
                 input_shape=(224, 224, 3),
                 input_tensor=None,
                 dropblock=None,
                 num_classes=1000,
                 drop_rate=0.5):
        super(RegNetY16GF, self).__init__("RegNetY16GF",
                                          w_0=200,
                                          w_a=106.23,
                                          w_m=2.48,
                                          group_w=112,
                                          use_se=True,
                                          se_ratio=0.25,
                                          bot_mul=1.0,
                                          depth=18,
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


@BACKBONES.register
class RegNetY32GF(AnyNet):
    def __init__(self, 
                 convolution="conv2d",
                 normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                 activation=dict(activation="relu"),
                 output_indices=(3, 4),
                 strides=(2, 2, 2, 2, 2),
                 dilation_rates=(1, 1, 1, 1, 1),
                 frozen_stages=(-1,),
                 input_shape=(224, 224, 3),
                 input_tensor=None,
                 dropblock=None,
                 num_classes=1000,
                 drop_rate=0.5):
        super(RegNetY32GF, self).__init__("RegNetY32GF",
                                          w_0=232,
                                          w_a=115.89,
                                          w_m=2.53,
                                          group_w=232,
                                          use_se=True,
                                          se_ratio=0.25,
                                          bot_mul=1.0,
                                          depth=20,
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


if __name__ == "__main__":
    regnet = RegNetX12GF()
    regnet.torch2h5("/home/bail/Downloads/RegNetX-12GF_dds_8gpu.pyth", 
                    "/home/bail/Workspace/pretrained_weights/RegNetX12GF.h5")

    with tf.io.gfile.GFile("/home/bail/Documents/pandas.jpg", "rb") as gf:
        images = tf.image.decode_jpeg(gf.read())

    images = tf.image.convert_image_dtype(images, tf.float32)
    images = tf.image.resize(images, (224, 224))[None]

    cls = regnet.model(images, training=False)
    cls = tf.nn.softmax(cls)
    print(tf.argmax(tf.squeeze(cls)))
    print(tf.reduce_max(cls))
    print(tf.nn.top_k(tf.squeeze(cls), k=5))
