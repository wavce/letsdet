import numpy as np
import tensorflow as tf
from ..common import ConvNormActBlock
from models.builder import BACKBONES
from models.backbones import Backbone
from core.layers import build_activation
from core.layers import build_convolution
from core.layers import build_normalization


def basic_block(inputs,
                shortcut,
                filters,
                strides=1, 
                dilation_rate=1, 
                expansion=2,
                cardinality=32,
                data_format="channels_last", 
                normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                activation=dict(activation="relu"),
                kernel_initializer="he_normal",
                trainable=True,
                name="basic_block"):
    x = ConvNormActBlock(filters=filters, 
                         kernel_size=3, 
                         strides=strides, 
                         padding="same", 
                         data_format=data_format, 
                         dilation_rate=dilation_rate if strides == 1 else 1, 
                         trainable=trainable, 
                         normalization=normalization,
                         activation=activation,
                         kernel_initializer=kernel_initializer,
                         name=name + "/conv1")(inputs)
    x = ConvNormActBlock(filters=filters, 
                         kernel_size=3, 
                         strides=1, 
                         padding="same", 
                         data_format=data_format, 
                         dilation_rate=dilation_rate, 
                         trainable=trainable, 
                         kernel_initializer=kernel_initializer,
                         normalization=normalization,
                         activation=None,
                         name=name + "/conv2")(x)

    if shortcut is None:
        shortcut = inputs
    x = tf.keras.layers.Add(name=name + "/add")([x, shortcut])
    if isinstance(activation, dict):
        act_name = activation["activation"] 
        x = build_activation(name=name + "/" + act_name, **activation)(x)
    else:
        x = build_activation(activation=activation, name=name + "/" + activation)(x)

    return x


def bottleneck(inputs,
               shortcut,
               filters,
               strides=1, 
               dilation_rate=1, 
               expansion=2,
               cardinality=32,
               data_format="channels_last", 
               normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
               activation=dict(activation="relu"),
               trainable=True,
               kernel_initializer="he_normal",
               name="bottleneck"):
    x = ConvNormActBlock(filters=filters // expansion, 
                         kernel_size=1, 
                         strides=1, 
                         data_format=data_format, 
                         dilation_rate=1, 
                         trainable=trainable, 
                         normalization=normalization,
                         activation=activation,
                         kernel_initializer=kernel_initializer,
                         name=name + "/conv1")(inputs)
     
    x = ConvNormActBlock(filters=filters // expansion, 
                         kernel_size=3, 
                         strides=strides, 
                         data_format=data_format, 
                         dilation_rate=dilation_rate if strides == 1 else 1, 
                         trainable=trainable, 
                         normalization=normalization,
                         kernel_initializer=kernel_initializer,
                         activation=activation,
                         name=name + "/conv2")(x)
   
    x = ConvNormActBlock(filters=filters, 
                         kernel_size=1, 
                         strides=1, 
                         data_format=data_format, 
                         dilation_rate=1, 
                         trainable=trainable, 
                         normalization=normalization,
                         activation=None,
                         kernel_initializer=kernel_initializer,
                         name=name + "/conv3")(x)
    if shortcut is None:
        shortcut = inputs
    x = tf.keras.layers.Add(name=name + "/add")([x, shortcut])
    if isinstance(activation, dict):
        act_name = activation["activation"] 
        x = build_activation(name=name + "/" + act_name, **activation)(x)
    else:
        x = build_activation(activation=activation, name=name + "/" + activation)(x)

    return x
        

def bottleneckx(inputs,
                shortcut,
                filters,
                strides=1, 
                dilation_rate=1,  
                expansion=2,
                cardinality=32,
                data_format="channels_last", 
                normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                activation=dict(activation="relu"),
                trainable=True,
                kernel_initializer="he_normal",
                name="bottleneckx"):

    neck_filters = filters * cardinality // 32
    x = ConvNormActBlock(filters=neck_filters, 
                         kernel_size=1, 
                         strides=1, 
                         data_format=data_format, 
                         dilation_rate=1, 
                         trainable=trainable, 
                         normalization=normalization,
                         activation=activation,
                         kernel_initializer=kernel_initializer,
                         name=name + "/conv1")(inputs)
     
    x = ConvNormActBlock(filters=neck_filters, 
                         kernel_size=3, 
                         strides=strides, 
                         groups=cardinality,
                         data_format=data_format, 
                         dilation_rate=dilation_rate if strides == 1 else 1, 
                         trainable=trainable, 
                         normalization=normalization,
                         kernel_initializer=kernel_initializer,
                         activation=activation,
                         name=name + "/conv2")(x)
   
    x = ConvNormActBlock(filters=filters, 
                         kernel_size=1, 
                         strides=1, 
                         data_format=data_format, 
                         dilation_rate=1, 
                         trainable=trainable, 
                         normalization=normalization,
                         activation=None,
                         kernel_initializer=kernel_initializer,
                         name=name + "/conv3")(x)
    if shortcut is None:
        shortcut = inputs

    x = tf.keras.layers.Add(name=name + "/add")([x, shortcut])
    if isinstance(activation, dict):
        act_name = activation["activation"] 
        x = build_activation(name=name + "/" + act_name, **activation)(x)
    else:
        x = build_activation(activation=activation, name=name + "/" + activation)(x)

    return x
   

def root(inputs, 
         filters, 
         kernel_size, 
         residual, 
         trainable=True, 
         data_format="channels_last", 
         normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
         activation=dict(activation="relu"),
         kernel_initializer="he_normal",
         name="root"):
       
    channel_axis = -1 if data_format == "channels_last" else 1
    x = tf.keras.layers.Concatenate(axis=channel_axis, name=name + "/cat")(inputs)
    x = ConvNormActBlock(filters=filters, 
                         kernel_size=kernel_size, 
                         strides=1, 
                         data_format=data_format, 
                         normalization=normalization,
                         activation=None,
                         trainable=trainable,
                         kernel_initializer=kernel_initializer, 
                         name=name + "/conv")(x)
    if residual:
        x = tf.keras.layers.Add(name=name + "/add")([x, inputs[0]])
    if isinstance(activation, dict):
        act_name = activation["activation"] 
        x = build_activation(name=name + "/" + act_name, **activation)(x)
    else:
        x = build_activation(activation=activation, name=name + "/" + activation)(x)

    return x


def tree(inputs,
         levels, 
         block_fn,
         in_filters,
         out_filters,
         children=None,
         strides=1,
         expansion=2,
         cardinality=32,
         normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
         activation=dict(activation="relu"),
         kernel_initializer="he_normal",
         level_root=False,
         root_dim=0,
         root_kernel_size=1,
         dilation_rate=1,
         root_residual=False,
         trainable=True,
         data_format="channels_last",
         name="tree"):
    children = [] if children is None else children
    if strides > 1:
        bottom = tf.keras.layers.MaxPool2D(
            strides, strides, data_format=data_format, name=name + "/downsample")(inputs)
    else:
        bottom = inputs
    if in_filters != out_filters:
        residual = ConvNormActBlock(filters=out_filters, 
                                    kernel_size=1,
                                    strides=1, 
                                    data_format=data_format,
                                    normalization=normalization,
                                    activation=None,
                                    kernel_initializer=kernel_initializer, 
                                    trainable=trainable,
                                    name=name + "/project")(bottom)
    else:
        residual = bottom
    
    if level_root:
        children.append(bottom)

    if root_dim == 0:
        root_dim = 2 * in_filters
    if level_root:
        root_dim += out_filters

    if levels == 1:
        x1 = block_fn(inputs,
                      residual,
                      filters=out_filters, 
                      strides=strides,
                      dilation_rate=dilation_rate,
                      normalization=normalization,
                      activation=activation,
                      expansion=expansion,
                      cardinality=cardinality,
                      kernel_initializer=kernel_initializer,
                      data_format=data_format,
                      trainable=trainable,
                      name=name + "/tree1")
    else:
        x1 = tree(inputs,
                  children=None,
                  expansion=expansion,
                  cardinality=cardinality,
                  levels=levels - 1,
                  strides=strides,
                  block_fn=block_fn,
                  in_filters=in_filters,
                  out_filters=out_filters,
                  root_dim=0,
                  trainable=trainable,
                  data_format=data_format,
                  normalization=normalization,
                  root_kernel_size=root_kernel_size,
                  root_residual=root_residual,
                  kernel_initializer=kernel_initializer,
                  name=name + "/tree1")
    if levels == 1:
        x2 = block_fn(x1,
                      None,
                      filters=out_filters,
                      strides=1,
                      expansion=expansion,
                      cardinality=cardinality,
                      dilation_rate=dilation_rate,
                      normalization=normalization,
                      kernel_initializer=kernel_initializer,
                      activation=activation,
                      data_format=data_format,
                      trainable=trainable,
                      name=name + "/tree2")

        x = root(inputs=[x2, x1, *children],
                 filters=out_filters,
                 kernel_size=root_kernel_size,
                 residual=root_residual,
                 data_format=data_format,
                 trainable=trainable,
                 normalization=normalization,
                 kernel_initializer=kernel_initializer,
                 name=name + "/root")
    else:
        children.append(x1)
        x = tree(x1,
                 children=children,
                 expansion=expansion,
                 cardinality=cardinality,
                 levels=levels - 1,
                 block_fn=block_fn,
                 in_filters=out_filters,
                 out_filters=out_filters,
                 root_dim=root_dim + out_filters,
                 root_kernel_size=root_kernel_size,
                 dilation_rate=dilation_rate,
                 root_residual=root_residual,
                 data_format=data_format,
                 normalization=normalization,
                 trainable=trainable,
                 kernel_initializer=kernel_initializer,
                 name=name + "/tree2")
    
    return x


class DLA(Backbone):
    def __init__(self, 
                 name,
                 levels, 
                 num_blocks,
                 block_fn=basic_block,
                 expansion=2,
                 cardinality=32,
                 output_indices=(-1, ), 
                 strides=(2, 2, 2, 2, 2), 
                 dilation_rates=(1, 1, 1, 1, 1), 
                 kernel_initializer="he_normal",
                 normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                 activation=dict(activation="relu"),
                 frozen_stages=(-1, ), 
                 input_shape=None, 
                 input_tensor=None, 
                 residual_root=False,
                 return_levels=False,
                 pool_size=7,
                 linear_root=False,
                 num_classes=1000,
                 drop_rate=0.5,
                 **kwargs):
        super(DLA, self).__init__(name=name,
                                  kernel_initializer=kernel_initializer,
                                  normalization=normalization, 
                                  activation=activation, 
                                  output_indices=output_indices, 
                                  strides=strides, 
                                  dilation_rates=dilation_rates, 
                                  frozen_stages=frozen_stages, 
                                  input_shape=input_shape, 
                                  input_tensor=input_tensor,
                                  drop_rate=drop_rate,
                                  num_classes=num_classes,
                                  **kwargs)
        self.num_blocks = num_blocks
        self.block_fn = block_fn
        self.residual_root = residual_root
        self.levels = levels
        self.return_levels = return_levels
        self.expansion = 2
        self.cardinality = cardinality

    def build_model(self):
        mean_std_shape = [1, 1, 1, 3] if self.data_format == "channels_last" else [1, 3, 1, 1]

        def _norm(inp):
            mean = tf.constant([0.485, 0.456, 0.406], inp.dtype, mean_std_shape) * 255.
            std = 1. / (tf.constant([0.229, 0.224, 0.225], inp.dtype, mean_std_shape) * 255.)

            return (inp - mean) * std

        x = tf.keras.layers.Lambda(_norm, name="norm_input")(self.img_input) 
        
        x = ConvNormActBlock(filters=self.num_blocks[0], 
                             kernel_size=7, 
                             strides=1, 
                             data_format=self.data_format, 
                             normalization=self.normalization,
                             activation=self.activation,
                             trainable= 1 not in self.frozen_stages,
                             kernel_initializer=self.kernel_initializer,
                             name="base_layer")(x)

        x = self._make_level(x, self.num_blocks[0],  self.levels[0], 1, 
                             self.dilation_rates[0], 1 not in self.frozen_stages, "level0")
        outputs = [x]
        x = self._make_level(x, self.num_blocks[1],  self.levels[1], self.strides[0], 
                             self.dilation_rates[1], 1 not in self.frozen_stages, "level1")
        outputs.append(x)
        x = tree(x,
                 children=None,
                 cardinality=self.cardinality,
                 expansion=self.expansion,
                 levels=self.levels[2], 
                 block_fn=self.block_fn, 
                 in_filters=self.num_blocks[1], 
                 out_filters=self.num_blocks[2], 
                 strides=self.strides[1],
                 dilation_rate=self.dilation_rates[1],
                 trainable=2 not in self.frozen_stages, 
                 normalization=self.normalization,
                 activation=self.activation,
                 kernel_initializer=self.kernel_initializer,
                 level_root=False,
                 data_format=self.data_format,
                 root_residual=self.residual_root,
                 name="level2")
        outputs.append(x)
        x = tree(x,
                 children=None,
                 cardinality=self.cardinality,
                 expansion=self.expansion,
                 levels= self.levels[3], 
                 block_fn=self.block_fn, 
                 in_filters=self.num_blocks[2], 
                 out_filters=self.num_blocks[3], 
                 strides=self.strides[2],
                 dilation_rate=self.dilation_rates[2],
                 trainable=3 not in self.frozen_stages, 
                 normalization=self.normalization,
                 activation=self.activation,
                 kernel_initializer=self.kernel_initializer,
                 level_root=True,
                 data_format=self.data_format,
                 root_residual=self.residual_root,
                 name="level3")
        outputs.append(x)
        x = tree(x,
                 children=None,
                 cardinality=self.cardinality,
                 expansion=self.expansion,
                 levels= self.levels[4], 
                 block_fn=self.block_fn, 
                 in_filters=self.num_blocks[3], 
                 out_filters=self.num_blocks[4], 
                 strides=self.strides[3],
                 dilation_rate=self.dilation_rates[3],
                 trainable=4 not in self.frozen_stages, 
                 normalization=self.normalization,
                 activation=self.activation,
                 kernel_initializer=self.kernel_initializer,
                 level_root=True,
                 data_format=self.data_format,
                 root_residual=self.residual_root,
                 name="level4")
        outputs.append(x)
        x = tree(x,
                 children=None,
                 cardinality=self.cardinality,
                 expansion=self.expansion,
                 levels= self.levels[5], 
                 block_fn=self.block_fn, 
                 in_filters=self.num_blocks[4], 
                 out_filters=self.num_blocks[5], 
                 strides=self.strides[4],
                 trainable=5 not in self.frozen_stages,
                 dilation_rate=self.dilation_rates[4], 
                 normalization=self.normalization,
                 activation=self.activation,
                 kernel_initializer=self.kernel_initializer,
                 level_root=True,
                 data_format=self.data_format,
                 root_residual=self.residual_root,
                 name="level5")
        outputs.append(x)

        if self.return_levels:
            x = tf.keras.layers.AvgPool2D((7, 7), data_format=self.data_format)(x)
            outputs = tf.keras.layers.Conv2D(
                self.num_classes, kernel_size=1, strides=1, data_format=self.data_format, name="fc")(x)
  
        return tf.keras.Model(inputs=self.img_input, outputs=outputs, name=self.name)
    
    def _make_level(self, inputs, filters, num_convs, strides=1, dilation_rate=1, trainable=True, name="level"):
        
        x = inputs
        for i in range(num_convs):
            x = ConvNormActBlock(filters=filters, 
                                 kernel_size=3, 
                                 strides=strides if i == 0 else 1, 
                                 dilation_rate=dilation_rate,
                                 normalization=self.normalization,
                                 activation=self.activation,
                                 kernel_initializer=self.kernel_initializer,
                                 trainable=trainable,
                                 data_format=self.data_format,
                                 name=name + "/%d" % i)(x)
        
        return x


@BACKBONES.register("DLA34")
def DLA34(normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
          activation=dict(activation="relu"),
          output_indices=(-1, ), 
          strides=(2, 2, 2, 2, 2), 
          dilation_rates=(1, 1, 1, 1, 1), 
          frozen_stages=(-1, ), 
          input_shape=None, 
          input_tensor=None, 
          dropblock=None,
          return_levels=True, 
          num_classes=1000,
          data_format="channels_last",
          drop_rate=0.5,
          **kwargs):
    return DLA(name="dla34",
               levels=(1, 1, 1, 2, 2, 1), 
               num_blocks=[16, 32, 64, 128, 256, 512],
               strides=strides,
               block_fn=basic_block,
               output_indices=output_indices, 
               dilation_rates=dilation_rates, 
               normalization=normalization,
               activation=activation,
               frozen_stages=frozen_stages, 
               input_shape=input_shape, 
               input_tensor=input_tensor, 
               residual_root=False,
               return_levels=return_levels,
               pool_size=7,
               linear_root=False,
               num_classes=num_classes,
               data_format=data_format,
               drop_rate=drop_rate,
               **kwargs).build_model()


@BACKBONES.register("DLA46C")
def DLA46C(normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
           activation=dict(activation="relu"),
           output_indices=(3, 4), 
           strides=(2, 2, 2, 2, 2), 
           dilation_rates=(1, 1, 1, 1, 1), 
           frozen_stages=(-1, ), 
           input_shape=None, 
           input_tensor=None, 
           return_levels=True, 
           dropblock=None, 
           num_classes=1000,
           data_format="channels_last",
           drop_rate=0.5,
           **kwargs):
    return DLA(name="dla46_c",
               levels=(1, 1, 1, 2, 2, 1), 
               num_blocks=[16, 32, 64, 64, 128, 256],
               strides=strides,
               expansion=2,
               block_fn=bottleneck,
               output_indices=output_indices, 
               dilation_rates=dilation_rates, 
               normalization=normalization,
               activation=activation,
               frozen_stages=frozen_stages, 
               input_shape=input_shape, 
               input_tensor=input_tensor, 
               residual_root=False,
               return_levels=return_levels,
               pool_size=7,
               linear_root=False,
               num_classes=num_classes,
               data_format=data_format,
               drop_rate=drop_rate,
               **kwargs).build_model()
                            

@BACKBONES.register("DLA46XC")
def DLA46XC(normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
            activation=dict(activation="relu"),
            output_indices=(-1, ), 
            strides=(2, 2, 2, 2, 2), 
            dilation_rates=(1, 1, 1, 1, 1), 
            frozen_stages=(-1, ), 
            input_shape=None, 
            input_tensor=None, 
            dropblock=None, 
            return_levels=True, 
            num_classes=1000,
            data_format="channels_last",
            drop_rate=0.5,
            **kwargs):

    return DLA(name="dla46x_c",
               levels=(1, 1, 1, 2, 3, 1), 
               num_blocks=[16, 32, 64, 64, 128, 256],
               strides=strides,
               block_fn=bottleneckx,
               expansion=2,
               output_indices=output_indices, 
               dilation_rates=dilation_rates, 
               normalization=normalization,
               activation=activation,
               frozen_stages=frozen_stages, 
               input_shape=input_shape, 
               input_tensor=input_tensor, 
               residual_root=False,
               return_levels=return_levels,
               pool_size=7,                
               linear_root=False,
               num_classes=num_classes,
               data_format=data_format,
               drop_rate=drop_rate,
               **kwargs).build_model()


@BACKBONES.register("DLA60C")
def DLA60C(normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
           activation=dict(activation="relu"),
           output_indices=(-1, ), 
           strides=(2, 2, 2, 2, 2), 
           dilation_rates=(1, 1, 1, 1, 1),  
           frozen_stages=(-1, ), 
           input_shape=None, 
           return_levels=True, 
           input_tensor=None, 
           dropblock=None, 
           num_classes=1000,
           drop_rate=0.5,
           **kwargs):
    
    return DLA(name="dla60_c",
               levels=(1, 1, 1, 2, 3, 1), 
               num_blocks=[16, 32, 64, 64, 128, 256],
               strides=strides,
               block_fn=bottleneck,
               expansion=2,
               output_indices=output_indices, 
               dilation_rates=dilation_rates, 
               normalization=normalization,
               activation=activation,
               frozen_stages=frozen_stages, 
               input_shape=input_shape, 
               input_tensor=input_tensor, 
               residual_root=False,
               return_levels=return_levels,
               pool_size=7,
               linear_root=False,
               num_classes=num_classes,
               drop_rate=drop_rate,
               **kwargs).build_model()


@BACKBONES.register("DLA60XC")
def DLA60XC(normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
            activation=dict(activation="relu"),
            output_indices=(3, 4), 
            strides=(2, 2, 2, 2, 2), 
            dilation_rates=(1, 1, 1, 1, 1),
            frozen_stages=(-1, ), 
            input_shape=None, 
            input_tensor=None, 
            dropblock=None, 
            return_levels=True, 
            num_classes=1000,
            drop_rate=0.5,
            **kwargs):

    return DLA(name="dla60x_c",
               levels=(1, 1, 1, 2, 3, 1), 
               num_blocks=[16, 32, 64, 64, 128, 256],
               strides=strides,
               block_fn=bottleneckx,
               expansion=2,
               output_indices=output_indices, 
               dilation_rates=dilation_rates, 
               normalization=normalization,
               activation=activation,
               frozen_stages=frozen_stages, 
               input_shape=input_shape, 
               input_tensor=input_tensor, 
               residual_root=False,
               return_levels=return_levels,
               pool_size=7,
               linear_root=False,
               num_classes=num_classes,
               drop_rate=drop_rate,
               **kwargs).build_model()


@BACKBONES.register("DLA60")
def DLA60(normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
          activation=dict(activation="relu"),
          output_indices=(3, 4), 
          strides=(2, 2, 2, 2, 2), 
          dilation_rates=(1, 1, 1, 1, 1),  
          frozen_stages=(-1, ), 
          input_shape=None, 
          input_tensor=None, 
          dropblock=None, 
          return_levels=True, 
          num_classes=1000,
          drop_rate=0.5,
          **kwargs):

    return DLA(name="dla60",
               levels=(1, 1, 1, 2, 3, 1), 
               num_blocks=[16, 32, 128, 256, 512, 1024],
               strides=strides,
               block_fn=bottleneck,
               expansion=2,
               output_indices=output_indices, 
               dilation_rates=dilation_rates, 
               normalization=normalization,
               activation=activation,
               frozen_stages=frozen_stages, 
               input_shape=input_shape, 
               input_tensor=input_tensor, 
               residual_root=False,
               return_levels=return_levels,
               pool_size=7,
               linear_root=False,
               num_classes=num_classes,                
               drop_rate=drop_rate,
               **kwargs).build_model()


@BACKBONES.register("DLA60X")
def DLA60X(normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
           activation=dict(activation="relu"),
           output_indices=(-1, ), 
           strides=(2, 2, 2, 2, 2), 
           dilation_rates=(1, 1, 1, 1, 1), 
           frozen_stages=(-1, ), 
           input_shape=None, 
           input_tensor=None, 
           dropblock=None, 
           return_levels=True, 
           num_classes=1000,
           drop_rate=0.5,
           **kwargs):

    return DLA(name="dla60x",
               levels=(1, 1, 1, 2, 3, 1), 
               num_blocks=[16, 32, 128, 256, 512, 1024],
               strides=strides,
               block_fn=bottleneckx,
               expansion=2,
               output_indices=output_indices, 
               dilation_rates=dilation_rates, 
               normalization=normalization,
               activation=activation,
               frozen_stages=frozen_stages, 
               input_shape=input_shape, 
               input_tensor=input_tensor, 
               residual_root=False,
               return_levels=return_levels,
               pool_size=7,
               linear_root=False,
               num_classes=num_classes,
               drop_rate=drop_rate,
               **kwargs).build_model()


@BACKBONES.register("DLA102")
def DLA102(normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
           activation=dict(activation="relu"),
           output_indices=(3, 4), 
           strides=(2, 2, 2, 2, 2), 
           dilation_rates=(1, 1, 1, 1, 1),  
           frozen_stages=(-1, ), 
           input_shape=None, 
           input_tensor=None, 
           return_levels=True, 
           dropblock=None, 
           num_classes=1000,
           drop_rate=0.5,
           **kwargs):

    return DLA(name="dla102",
               levels=(1, 1, 1, 3, 4, 1), 
               num_blocks=[16, 32, 128, 256, 512, 1024],
               strides=strides,
               expansion=2,
               block_fn=bottleneck,
               output_indices=output_indices, 
               dilation_rates=dilation_rates, 
               normalization=normalization,
               activation=activation,
               frozen_stages=frozen_stages, 
               input_shape=input_shape, 
               input_tensor=input_tensor, 
               residual_root=True,
               return_levels=return_levels,
               pool_size=7,
               linear_root=True,
               num_classes=num_classes,
               drop_rate=drop_rate,
               **kwargs).build_model()


@BACKBONES.register("DLA102X")
def DLA102X(normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
            activation=dict(activation="relu"),
            output_indices=(-1, ), 
            strides=(2, 2, 2, 2, 2), 
            dilation_rates=(1, 1, 1, 1, 1),  
            frozen_stages=(-1, ), 
            input_shape=None, 
            input_tensor=None, 
            dropblock=None, 
            return_levels=True, 
            num_classes=1000,
            drop_rate=0.5,
            **kwargs):
 
    return DLA(name="dla102x",
               levels=(1, 1, 1, 3, 4, 1), 
               num_blocks=[16, 32, 128, 256, 512, 1024],
               strides=strides,
               block_fn=bottleneckx,
               expansion=2,
               output_indices=output_indices, 
               dilation_rates=dilation_rates, 
               normalization=normalization,
               activation=activation,
               frozen_stages=frozen_stages, 
               input_shape=input_shape, 
               input_tensor=input_tensor, 
               residual_root=True,
               return_levels=return_levels,
               pool_size=7,
               linear_root=False,
               num_classes=num_classes,
               drop_rate=drop_rate,
               **kwargs).build_model()


@BACKBONES.register("DLA102X2")
def DLA102X2(normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
             activation=dict(activation="relu"),
             output_indices=(-1, ), 
             strides=(2, 2, 2, 2, 2), 
             dilation_rates=(1, 1, 1, 1, 1),  
             frozen_stages=(-1, ), 
             input_shape=None, 
             input_tensor=None, 
             dropblock=None, 
             return_levels=True, 
             num_classes=1000,
             drop_rate=0.5,
             **kwargs):
    
    return DLA(name="dla102x2",
               levels=(1, 1, 1, 3, 4, 1), 
               num_blocks=[16, 32, 128, 256, 512, 1024],
               strides=strides,
               block_fn=bottleneck,
               expansion=2, 
               cardinality=64,
               output_indices=output_indices, 
               dilation_rates=dilation_rates, 
               normalization=normalization,
               activation=activation,
               frozen_stages=frozen_stages, 
               input_shape=input_shape, 
               input_tensor=input_tensor, 
               residual_root=True,
               return_levels=return_levels,
               pool_size=7,
               linear_root=False,
               num_classes=num_classes,
               drop_rate=drop_rate,
               **kwargs).build_model()


@BACKBONES.register("DLA169")
def DLA169(convolution='conv2d', 
           normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
           activation=dict(activation="relu"),
           output_indices=(3, 4), 
           strides=(2, 2, 2, 2, 2), 
           dilation_rates=(1, 1, 1, 1, 1),  
           frozen_stages=(-1, ), 
           input_shape=None, 
           input_tensor=None, 
           dropblock=None, 
           return_levels=True, 
           num_classes=1000,
           drop_rate=0.5,
           **kwargs):

    return DLA(name="dla102x2",
               levels=(1, 1, 2, 3, 5, 1), 
               num_blocks=[16, 32, 128, 256, 512, 1024],
               strides=strides,
               block_fn=bottleneck,
               expansion=2,
               output_indices=output_indices, 
               dilation_rates=dilation_rates, 
               normalization=normalization,
               activation=activation,
               frozen_stages=frozen_stages, 
               input_shape=input_shape, 
               input_tensor=input_tensor, 
               residual_root=True,
               return_levels=return_levels,
               pool_size=7,
               linear_root=False,
               num_classes=num_classes,
               drop_rate=drop_rate,
               **kwargs).build_model()


def _torch2h5(model, torch_weights_path):
    import re
    import torch
    import numpy as np
    import torch.nn as nn

    # model.summary()
    t_dla = torch.load(torch_weights_path)
    # for k, v in t_dla.items():
    #     print(k, v.shape)
    
    for weight in model.weights:
        name = weight.name
        # print(name, weight.shape)
        name = name.split(":")[0]
        name = name.replace("/", ".")
        name = name.replace("frozen_batch_norm", "batch_norm")

        if "base_layer.conv2d.kernel" in name:
            name = name.replace("conv2d.kernel", "0.weight")
        if "base_layer.batch_norm" in name:
            name = name.replace("batch_norm", "1")
        if "level0.0.conv2d.kernel" in name:
            name = name.replace("conv2d.kernel", "weight")
        if "level0.0.batch_norm" in name:
            name = name.replace("0.batch_norm", "1")
        if "level1.0.conv2d.kernel" in name:
            name = name.replace("conv2d.kernel", "weight")
        if "level1.0.batch_norm" in name:
            name = name.replace("0.batch_norm", "1")
        for i in range(3):
            if "conv%d.batch_norm" % (i + 1) in name:
                name = name.replace("conv%d.batch_norm" % (i + 1), "bn%d" % (i + 1))
        if "project.conv2d" in name:
            name = name.replace("project.conv2d.kernel", "project.0.weight")
        if "project.batch_norm" in name:
            name = name.replace("project.batch_norm", "project.1")
        if "root.conv.batch_norm" in name:
            name = name.replace("root.conv.batch_norm", "root.bn")
        if "conv2d.kernel" in name:
            name = name.replace("conv2d.kernel", "weight")
        if "gamma" in name:
            name = name.replace("gamma", "weight")
        if "beta" in name:
            name = name.replace("beta", "bias")
        if "moving_mean" in name:
            name = name.replace("moving_mean", "running_mean")
        if "moving_variance" in name:
            name = name.replace("moving_variance", "running_var")
        if "kernel" in name:
            name = name.replace("kernel", "weight")
        name = re.sub("_\d+\d*", "", name)

        # print(name)
        tw = t_dla[name].numpy()
        if len(tw.shape) == 4:
            tw = np.transpose(tw, (2, 3, 1, 0))
        weight.assign(tw)
    

if __name__ == "__main__":
    name = "dla34"
    model = DLA34(input_shape=(3, 224, 224), 
                  normalization=dict(normalization="frozen_batch_norm", momentum=0.9, epsilon=1e-5, axis=1, trainable=True),
                  output_indices=(-1, ), 
                  data_format="channels_first")
    _torch2h5(model, 
             "/home/bail/Data/data2/pretrained_weights/%s.pth" % name)

    with tf.io.gfile.GFile("/home/bail/Documents/pandas.jpg", "rb") as gf:
        images = tf.image.decode_jpeg(gf.read())

    images = tf.cast(images, tf.float32)
    images = tf.image.resize(images, (224, 224))[None]

    images = tf.transpose(images, [0, 3, 1, 2])
    logits = model(images, training=False)
    probs = tf.nn.softmax(logits, 1)
    print(tf.nn.top_k(tf.squeeze(probs), k=5))

