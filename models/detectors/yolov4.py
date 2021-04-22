import math
import numpy as np
import tensorflow as tf
from core import build_nms
from core import build_loss
from utils import box_utils
from .detector import Detector
from models.builder import DETECTORS
from ..common import ConvNormActBlock
from core.layers import build_activation
from core.layers import build_normalization
from core.losses.iou_loss import compute_iou


def bottleneck(inputs,
               filters, 
               dilation_rate=1,
               expansion=0.5,
               data_format="channels_last",
               normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-3, axis=-1, trainable=True),
               activation=dict(activation="leaky_relu", alpha=0.1),
               kernel_initializer="glorot_uniform",
               trainable=True,
               shortcut=True,
               index=0):
    channel_axis = -1 if data_format == "channels_last" else 1
    normalization["axis"] = channel_axis

    x = ConvNormActBlock(filters=int(filters * expansion),
                         kernel_size=1,
                         dilation_rate=1,
                         normalization=normalization,
                         activation=activation,
                         trainable=trainable,
                         kernel_initializer=kernel_initializer,
                         name="conv" + str(index))(inputs)
    x = ConvNormActBlock(filters=filters,
                         kernel_size=3,
                         dilation_rate=dilation_rate,
                         normalization=normalization,
                         activation=activation,
                         trainable=trainable,
                         kernel_initializer=kernel_initializer,
                         name="conv" + str(index + 1))(x)
    if shortcut:
        x = tf.keras.layers.Add(name="add%d" % index)([x, inputs])

    return x, index + 1


def bottleneck_csp(inputs,
                   n, 
                   filters, 
                   strides=1, 
                   dilation_rate=1, 
                   expansion=0.5, 
                   data_format="channels_last", 
                   normalization=dict(normalization="batch_norm", momentum=0.997, epsilon=1e-4, trainable=False), 
                   activation="mish", 
                   kernel_initializer="glorot_uniform", 
                   trainable=True, 
                   shortcut=True,
                   index=0):
    channel_axis = -1 if data_format == "channels_last" else 1
    normalization["axis"] = channel_axis

    x = ConvNormActBlock(filters=filters,
                         kernel_size=3,
                         strides=strides,
                         dilation_rate=dilation_rate,
                         trainable=trainable,
                         kernel_initializer=kernel_initializer,
                         normalization=normalization,
                         activation=activation,
                         data_format=data_format,
                         name="conv%d" % index)(inputs)
    index += 1
    midfilters = filters if filters == 64 else int(filters * expansion)
    route = ConvNormActBlock(filters=midfilters,
                             kernel_size=1,
                             strides=1,
                             dilation_rate=dilation_rate,
                             trainable=trainable,
                             kernel_initializer=kernel_initializer,
                             normalization=normalization,
                             activation=activation,
                             data_format=data_format,
                             name="conv%d" % index)(x)
    index += 1
    x = ConvNormActBlock(filters=midfilters,
                         kernel_size=1,
                         strides=1,
                         dilation_rate=dilation_rate,
                         trainable=trainable,
                         kernel_initializer=kernel_initializer,
                         normalization=normalization,
                         activation=activation,
                         data_format=data_format,
                         name="conv%d" % index)(x)
    index += 1
    
    for _ in range(n):
        x, index = bottleneck(inputs=x, 
                              filters=midfilters, 
                              dilation_rate=dilation_rate,
                              expansion=0.5 if filters == 64 else 1, 
                              data_format=data_format, 
                              normalization=normalization, 
                              activation=activation, 
                              kernel_initializer=kernel_initializer, 
                              trainable=trainable, 
                              shortcut=shortcut, 
                              index=index)
        index += 1

    x = ConvNormActBlock(filters=midfilters,
                         kernel_size=1,
                         strides=1,
                         dilation_rate=dilation_rate,
                         trainable=trainable,
                         kernel_initializer=kernel_initializer,
                         normalization=normalization,
                         activation=activation,
                         data_format=data_format,
                         name="conv%d" % index)(x)
    
    index += 1
    channels_axis = -1 if data_format == "channels_last" else 1
    x = tf.keras.layers.Concatenate(axis=channels_axis, name="cat%d" % index)([x, route])
    x = ConvNormActBlock(filters=filters,
                         kernel_size=1,
                         strides=1,
                         dilation_rate=dilation_rate,
                         trainable=trainable,
                         kernel_initializer=kernel_initializer,
                         normalization=normalization,
                         activation=activation,
                         data_format=data_format,
                         name="conv%d" % index)(x)

    return x, index


def bottleneck2(inputs,
                filters, 
                dilation_rate=1,
                expansion=0.5,
                groups=1,
                data_format="channels_last",
                normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-3, axis=-1, trainable=True),
                activation=dict(activation="leaky_relu", alpha=0.1),
                kernel_initializer="glorot_uniform",
                trainable=True,
                shortcut=True,
                name="bottleneck_v5"):
    channel_axis = -1 if data_format == "channels_last" else 1
    normalization["axis"] = channel_axis

    midfilters = int(filters * expansion)
    x = ConvNormActBlock(filters=midfilters, 
                         kernel_size=1, 
                         strides=1, 
                         data_format=data_format,
                         kernel_initializer=kernel_initializer,
                         trainable=trainable,
                         normalization=normalization,
                         activation=activation,
                         name=name + "/cv1")(inputs)
    x = ConvNormActBlock(filters=filters,
                         kernel_size=3,
                         strides=1,
                         groups=groups,
                         normalization=normalization,
                         activation=activation,
                         name=name + "/cv2")(x)
    if shortcut:
        x = tf.keras.layers.Add(name=name + "/add")([x, inputs])
    return x


def bottleneck_csp2(inputs,
                    n, 
                    filters, 
                    strides=1, 
                    dilation_rate=1, 
                    expansion=0.5, 
                    groups=1,
                    data_format="channels_last", 
                    normalization=dict(normalization="batch_norm", momentum=0.997, epsilon=1e-4, trainable=False), 
                    activation="mish", 
                    kernel_initializer="glorot_uniform", 
                    trainable=True, 
                    shortcut=True,
                    name="bottleneck_csp_v5"):
    channel_axis = -1 if data_format == "channels_last" else 1
    normalization["axis"] = channel_axis

    midfilters = int(filters * expansion)
    x1 = ConvNormActBlock(filters=midfilters,
                          kernel_size=1,
                          strides=1,
                          trainable=trainable,
                          kernel_initializer=kernel_initializer,
                          normalization=normalization,
                          activation=activation,
                          data_format=data_format,
                          name=name + "/cv1")(inputs)
    for i in range(n):
        x1 = bottleneck2(inputs=x1, 
                         filters=midfilters, 
                         dilation_rate=dilation_rate,
                         expansion=1., 
                         groups=groups,
                         data_format=data_format, 
                         normalization=normalization, 
                         activation=activation, 
                         kernel_initializer=kernel_initializer, 
                         trainable=trainable, 
                         shortcut=shortcut, 
                         name=name + "/m/%d" % i)
    x1 = tf.keras.layers.Conv2D(filters=midfilters,
                                kernel_size=1,
                                strides=1,
                                use_bias=False,
                                kernel_initializer=kernel_initializer,
                                name=name + "/cv3")(x1)
    x2 = tf.keras.layers.Conv2D(filters=midfilters,
                                kernel_size=1,
                                strides=1,
                                use_bias=False,
                                kernel_initializer=kernel_initializer,
                                name=name + "/cv2")(inputs)

    x = tf.keras.layers.Concatenate(axis=channel_axis, name=name + "/cat")([x1, x2])
    x = build_normalization(name=name + "/bn", **normalization)(x)
    x = build_activation(activation=activation, name=name + "/" + activation)(x)
    x = ConvNormActBlock(filters=filters,
                         kernel_size=1,
                         strides=1,
                         data_format=data_format,
                         kernel_initializer=kernel_initializer,
                         trainable=trainable,
                         normalization=normalization,
                         activation=activation,
                         name=name + "/cv4")(x)
    return x
    

def bottleneck_csp2_2(inputs,
                      n, 
                      filters, 
                      strides=1, 
                      dilation_rate=1, 
                      expansion=0.5, 
                      data_format="channels_last", 
                      normalization=dict(normalization="batch_norm", momentum=0.997, epsilon=1e-4, trainable=False), 
                      activation="mish", 
                      kernel_initializer="glorot_uniform", 
                      trainable=True, 
                      shortcut=False,
                      name="bottleneck_csp_2"):
    channel_axis = -1 if data_format == "channels_last" else 1
    normalization["axis"] = channel_axis

    midfilters = int(filters)
    x = ConvNormActBlock(filters=midfilters,
                         kernel_size=1,
                         strides=1,
                         trainable=trainable,
                         kernel_initializer=kernel_initializer,
                         normalization=normalization,
                         activation=activation,
                         data_format=data_format,
                         name=name + "/cv1")(inputs)
    x1 = x
    for i in range(n):
        x1 = bottleneck2(inputs=x1, 
                         filters=midfilters, 
                         dilation_rate=dilation_rate,
                         expansion=1., 
                         data_format=data_format, 
                         normalization=normalization, 
                         activation=activation, 
                         kernel_initializer=kernel_initializer, 
                         trainable=trainable, 
                         shortcut=shortcut, 
                         name=name + "/m/%d" % i)
    x2 = tf.keras.layers.Conv2D(filters=midfilters,
                                kernel_size=1,
                                strides=1,
                                use_bias=False,
                                kernel_initializer=kernel_initializer,
                                name=name + "/cv2")(x)

    x = tf.keras.layers.Concatenate(axis=channel_axis, name=name + "/cat")([x1, x2])
    normalization["trainable"] = trainable
    x = build_normalization(name=name + "/bn", **normalization)(x)
    x = build_activation(activation=activation, name=name + "/" + activation)(x)
    x = ConvNormActBlock(filters=filters,
                         kernel_size=1,
                         strides=1,
                         data_format=data_format,
                         kernel_initializer=kernel_initializer,
                         trainable=trainable,
                         normalization=normalization,
                         activation=activation,
                         name=name + "/cv3")(x)
    return x


def spatial_pyramid_pooling_csp(inputs, 
                               filters, 
                               pool_sizes=[5, 9, 13],
                               strides=1, 
                               dilation_rate=1, 
                               expansion=0.5, 
                               groups=1,
                               data_format="channels_last", 
                               normalization=dict(normalization="batch_norm", momentum=0.997, epsilon=1e-4, trainable=False), 
                               activation="mish", 
                               kernel_initializer="glorot_uniform", 
                               trainable=True, 
                               shortcut=True,
                               name="sppcsp"):
    channel_axis = -1 if data_format == "channels_last" else 1
    normalization["axis"] = channel_axis

    midfilters = int(2 * filters * expansion)
    x1 = ConvNormActBlock(filters=midfilters,
                          kernel_size=1,
                          strides=1,
                          data_format=data_format,
                          kernel_initializer=kernel_initializer,
                          trainable=trainable,
                          normalization=normalization,
                          activation=activation,
                          name=name + "/cv1")(inputs)
    x1 = ConvNormActBlock(filters=midfilters,
                          kernel_size=3,
                          strides=1,
                          data_format=data_format,
                          kernel_initializer=kernel_initializer,
                          trainable=trainable,
                          normalization=normalization,
                          activation=activation,
                          name=name + "/cv3")(x1) 
    x1 = ConvNormActBlock(filters=midfilters,
                          kernel_size=1,
                          strides=1,
                          data_format=data_format,
                          kernel_initializer=kernel_initializer,
                          trainable=trainable,
                          normalization=normalization,
                          activation=activation,
                          name=name + "/cv4")(x1)
    p  = [x1]
    for i, k in enumerate(pool_sizes):
        p.append(tf.keras.layers.MaxPool2D(k, 1, "same", name=name + "/m/%d" % i)(x1))
    
    x1 = tf.keras.layers.Concatenate(axis=channel_axis, name=name + "/cat1")(p)
    x1 = ConvNormActBlock(filters=midfilters,
                          kernel_size=1,
                          strides=1,
                          data_format=data_format,
                          kernel_initializer=kernel_initializer,
                          trainable=trainable,
                          normalization=normalization,
                          activation=activation,
                          name=name + "/cv5")(x1) 
    x1 = ConvNormActBlock(filters=midfilters,
                          kernel_size=3,
                          strides=1,
                          data_format=data_format,
                          kernel_initializer=kernel_initializer,
                          trainable=trainable,
                          normalization=normalization,
                          activation=activation,
                          name=name + "/cv6")(x1)

    x2 = tf.keras.layers.Conv2D(filters=midfilters,
                                kernel_size=1,
                                data_format=data_format,
                                use_bias=False,
                                kernel_initializer=kernel_initializer,
                                name=name + "/cv2")(inputs)
    x = tf.keras.layers.Concatenate(axis=channel_axis, name=name + "/cat2")([x1, x2])
    normalization["trainable"] = trainable
    x = build_normalization(name=name + "/bn", **normalization)(x)
    x = build_activation(activation=activation, name=name + "/" + activation)(x)
    x = ConvNormActBlock(filters=filters,
                         kernel_size=1,
                         strides=1,
                         data_format=data_format,
                         kernel_initializer=kernel_initializer,
                         trainable=trainable,
                         name=name + "/cv7")(x)
    
    return x 


def spatial_pyramid_pooling2(inputs,
                             filters, 
                             pool_sizes=[5, 9, 13],
                             strides=1, 
                             dilation_rate=1, 
                             expansion=0.5, 
                             groups=1,
                             data_format="channels_last", 
                             normalization=dict(normalization="batch_norm", momentum=0.997, epsilon=1e-4, trainable=False), 
                             activation="mish", 
                             kernel_initializer="glorot_uniform", 
                             trainable=True, 
                             shortcut=True,
                             name="spp"):
    midfilters = int(filters * expansion)
    x = ConvNormActBlock(filters=midfilters,
                         kernel_size=1,
                         strides=1,
                         data_format=data_format,
                         kernel_initializer=kernel_initializer,
                         trainable=trainable,
                         normalization=normalization,
                         activation=activation,
                         name=name + "/cv1")(inputs) 
    p  = [x]
    for i, k in enumerate(pool_sizes):
        p.append(tf.keras.layers.MaxPool2D(k, 1, "same", name=name + "/m/%d" % i)(x))
    
    channel_axis = -1 if data_format == "channels_last" else 1
    x = tf.keras.layers.Concatenate(axis=channel_axis, name=name + "/cat")(p)
    x = ConvNormActBlock(filters=filters,
                         kernel_size=1,
                         strides=1,
                         data_format=data_format,
                         kernel_initializer=kernel_initializer,
                         trainable=trainable,
                         normalization=normalization,
                         activation=activation,
                         name=name + "/cv2")(x) 

    return x


class Focus(tf.keras.Model):
    def __init__(self, 
                 filters, 
                 kernel_size=1, 
                 strides=1, 
                 groups=1, 
                 data_format="channels_last", 
                 normalization=dict(normalization="batch_norm", momentum=0.997, epsilon=1e-4, trainable=False), 
                 activation="mish", 
                 kernel_initializer="glorot_uniform", 
                 trainable=True, 
                 name="focus"):
        super(Focus, self).__init__(name=name)
        
        self.conv = ConvNormActBlock(filters=filters, 
                                     kernel_size=kernel_size, 
                                     strides=strides, 
                                     groups=groups, 
                                     padding="same",
                                     data_format=data_format,
                                     activation=activation, 
                                     normalization=normalization,
                                     kernel_initializer=kernel_initializer,
                                     trainable=trainable,
                                     name="conv")
        
    def call(self, inputs, training=None):
        x = tf.concat([inputs[:, ::2, ::2, :], 
                       inputs[:, 1::2, ::2, :], 
                       inputs[:, ::2, 1::2, :], 
                       inputs[:, 1::2, 1::2, :]], axis=-1)
        x = self.conv(x, training=training)
       
        return x

            
class Detect(tf.keras.Model):
    def __init__(self, num_classes=80, anchors=(), strides=(8, 16, 32), name="detect"):
        super(Detect, self).__init__(name=name)
        
        self.num_classes = num_classes
        self.num_outputs = num_classes + 5
        
        self.num_layers = len(anchors)  # number of detection layers
        self.num_anchors = len(anchors[0]) // 2 # number of anchors per layer
        self.anchors = anchors
        self.strides = strides
        
        biases = []
        for i in range(self.num_layers):
            s = strides[i]
            bias = np.zeros([self.num_anchors, self.num_outputs])
            bias[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            bias[:, 5:] += math.log(0.6 / (num_classes - 0.99)) # cls
            bias = np.reshape(bias, [-1])
            biases.append(bias)
        self.m = [
            tf.keras.layers.Conv2D(
                filters=self.num_anchors * self.num_outputs, 
                kernel_size=1, 
                bias_initializer=tf.keras.initializers.Constant(biases[i]),
                name="predicted/%d" % i) 
            for i in range(self.num_layers)
        ]
    
    def call(self, inputs):
        outputs = []
        for i in range(self.num_layers):
            x = self.m[i](inputs[i])
            shape = tf.shape(x)
            bs, ny, nx = shape[0], shape[1], shape[2]
            x = tf.reshape(x, [bs, ny, nx, self.num_anchors, self.num_outputs])
            
            outputs.append(x)

        return outputs


def spatial_pyramid_pooling(inputs, pool_sizes=[5, 9, 13], data_format="channels_last"):
    outputs = [inputs]
    for ps in pool_sizes:
        outputs.append(tf.keras.layers.MaxPool2D(ps, strides=1, padding="same")(inputs))
    
    axis = -1 if data_format == "channels_last" else 1

    return tf.keras.layers.Concatenate(axis=axis, name="spp_cat")(outputs[::-1])


def bottleneck_csp_tiny(inputs, 
                        filters,
                        n=1,
                        groups=2,
                        group_id=1,
                        data_format="channels_last", 
                        normalization=dict(normalization="batch_norm", momentum=0.997, epsilon=1e-4, trainable=False), 
                        activation=dict(activation="leaky_relu", alpha=0.1), 
                        kernel_initializer="glorot_uniform", 
                        trainable=True,
                        index=0):
    x = ConvNormActBlock(filters=filters,
                         kernel_size=3,
                         data_format=data_format,
                         kernel_initializer=kernel_initializer,
                         activation=activation,
                         trainable=trainable,
                         name="conv%d" % index)(inputs)
    
    axis = -1 if data_format == "channels_last" else 1
    x_groups = tf.keras.layers.Lambda(lambda x: tf.split(x, groups, axis), name="split%d" % index)(x)

    x1 = x_groups[group_id]
    for _ in range(n):
        index += 1
        x1 = ConvNormActBlock(filters=filters // groups, 
                              kernel_size=3,
                              data_format=data_format,
                              kernel_initializer=kernel_initializer,
                              trainable=trainable,
                              normalization=normalization,
                              activation=activation,
                              name="conv%d" % index)(x1)
        shortcut = x1
        index += 1
        x1 = ConvNormActBlock(filters=filters // groups, 
                              data_format=data_format,
                              kernel_size=3,
                              kernel_initializer=kernel_initializer,
                              trainable=trainable,
                              normalization=normalization,
                              activation=activation,
                              name="conv%d" % index)(x1)
        x1 = tf.keras.layers.Concatenate(axis=axis, name="cat%d" % index)([x1, shortcut])

    index += 1
    x1 = ConvNormActBlock(filters=filters, 
                          kernel_size=1,
                          kernel_initializer=kernel_initializer,
                          trainable=trainable,
                          data_format=data_format,
                          normalization=normalization,
                          activation=activation,
                          name="conv%d" % index)(x1)
    x = tf.keras.layers.Concatenate(axis=axis, name="cat%d" % index)([x, x1])

    return x, x1, index


@DETECTORS.register
class YOLOv4(Detector):
    def __init__(self, cfg, **kwargs):
        super(YOLOv4, self).__init__(cfg, **kwargs)

        self.g = 0.5
        self.off = tf.constant([[0, 0],
                                [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                                # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                                ], tf.float32) * self.g  # offsets
        
        self.nms = build_nms(**cfg.test.as_dict())
        
        self.num_anchors = self.cfg.num_anchors
        self.num_classes = self.cfg.num_classes
            
        assert "IoU" in cfg.bbox_loss.loss, "Only surpport IoU loss (e.g. IoULoss, GIoU, DIoU, CIoU)."
        self.bbox_loss_func = build_loss(**cfg.bbox_loss.as_dict())
        self.label_loss_func = build_loss(**cfg.label_loss.as_dict())
        self.conf_loss_func = build_loss(**cfg.conf_loss.as_dict())

        self.min_level = cfg.min_level
        self.max_level = cfg.max_level

        self.detector = self._create_model() 
        # self.model.summary()
    
    def load_pretrained_weights(self, pretrained_weights_path=None):
        self.detector.load_weights(pretrained_weights_path, by_name=True, skip_mismatch=True)
    
    def _create_model(self):
        
        def make_divisable(x, divisor):
            return math.ceil(x / divisor) * divisor

        print("\n%3s%18s%3s  %-40s%-32s" % ("", "from", "n", "module", "arguments"))
        index = 0
        
        input_shape = (list(self.cfg.input_size) + [3] 
                       if self.cfg.data_format == "channels_last" 
                       else [3] + list(self.cfg.input_size))
        inputs = tf.keras.layers.Input(shape=input_shape)
        x = tf.keras.layers.Lambda(lambda inp: inp / 255., name="norm_inputs")(inputs)
        layers = [x]
        outputs = []
        anchors = self.cfg.anchors
        num_classes = self.cfg.num_classes
        for i, (prev, n, unit_name, kwargs) in enumerate(self.cfg.model):   # from, number, module, args
            n = max(round(n * self.cfg.depth_multiplier), 1) if n > 1 else n  # depth gain
            if unit_name in ["ConvNormActBlock", "Bottleneck2", "SPPCSP", "Focus",
                             "CrossConv", "BottleneckCSP2", "Bottleneck", "BottleneckCSP2_2"]:
                kwargs["filters"] = make_divisable(kwargs["filters"] * self.cfg.width_multiplier, 8)

            print("%3s%18s%3s  %-40s%-32s" % (i, prev, n, unit_name, kwargs))
            if unit_name == "Focus":
                x = Focus(name=str(i), **kwargs)(layers[prev])
                layers.append(x)

            if unit_name == "BottleneckCSP":
                index += 1
                x, index = bottleneck_csp(layers[prev], n, index=index, **kwargs)
                layers.append(x)
            
            if unit_name == "BottleneckCSP2":
                x = bottleneck_csp2(layers[prev], n, name=str(i), **kwargs)
                layers.append(x)
            
            if unit_name == "BottleneckCSP2_2":
                index += 1
                x = bottleneck_csp2_2(layers[prev], n, name=str(i), **kwargs)
                layers.append(x)
            
            if unit_name == "SPPCSP":
                index += 1
                x = spatial_pyramid_pooling_csp(layers[prev], name=str(i), **kwargs)
                layers.append(x)
            
            if unit_name == "SPP":
                index += 1
                x = spatial_pyramid_pooling2(layers[prev], name=str(i), **kwargs)
                layers.append(x)

            if unit_name == "BottleneckCSPTiny":
                index += 1
                x, x1, index = bottleneck_csp_tiny(layers[prev], n=n, index=index, **kwargs)
                layers.append(x)

            if unit_name == "ConvNormActBlock":
                x = layers[prev + 1 if prev > 1 else prev]
                for _ in range(n):
                    index += 1
                    prev = prev + 1 if prev > 1 else prev
                    name = "conv%d" % index if "ScaledYOLOv4" not in self.cfg.detector else str(i)
                    x = ConvNormActBlock(name=name, **kwargs)(x)
                layers.append(x)
                
            if unit_name == "SpatialPyramidPooling":
                x = spatial_pyramid_pooling(x, **kwargs)
                layers.append(x)

            if unit_name == "Bottleneck":
                x = layers[prev]
                for _ in range(n):
                    index += 1
                    x, index = bottleneck(x, index=index, **kwargs)
                layers.append(x)

            if unit_name == "Upsample":
                x = tf.keras.layers.UpSampling2D(**kwargs)(layers[prev + 1 if prev > 0 else prev])
                layers.append(x)

            if unit_name == "Concat":
                x = tf.keras.layers.Concatenate(**kwargs)([layers[ii if ii < 0 else ii + 1] for ii in prev])
                layers.append(x)

            if unit_name == "Concat2":
                x = tf.keras.layers.Concatenate(**kwargs)([layers[prev], x1])
                layers.append(x)
            
            if unit_name == "MaxPool":
                x = tf.keras.layers.MaxPool2D(**kwargs)(layers[prev])
                layers.append(x)

            if unit_name == "Conv":
                index += 1
                x = tf.keras.layers.Conv2D(
                    filters=(5 + self.cfg.num_classes) * self.cfg.num_anchors, 
                    name="predicted%d" % index, **kwargs)(layers[prev + 1 if prev > 0 else prev])
                outputs.append(x)
                layers.append(x)
            
            if unit_name == "Detect":
                index += 1
                anchors = eval(kwargs["anchors"])
                num_classes = eval(kwargs["num_classes"])

                outputs = Detect(anchors=anchors, 
                                 num_classes=num_classes, 
                                 strides=self.cfg.strides, 
                                 name=str(i))([layers[l] for l in prev])
        
        if not self.training:
            outputs = tf.keras.layers.Lambda(self.get_boxes, name="GetBoxes")(outputs)
        return tf.keras.Model(inputs=inputs, outputs=outputs)

    def _calc_iou(self, tgt_boxes, anchor_wh):
        with tf.name_scope("anchor_iou"):
            tgt_boxes = tf.expand_dims(tgt_boxes, 1)
            anchor_wh = tf.expand_dims(anchor_wh, 0)
            tgt_wh = tgt_boxes[..., 2:4] - tgt_boxes[..., 0:2]
            min_wh = tf.math.minimum(tgt_wh, anchor_wh)
            
            union_area = tf.math.reduce_prod(tgt_wh, -1) + tf.math.reduce_prod(anchor_wh, -1)
            inter_area = tf.math.reduce_prod(min_wh, -1)
            
            return inter_area / (union_area - inter_area) 

    def get_targets(self, gt_boxes, gt_labels, grid_shape, anchors, strides):
        with tf.name_scope("get_targets_per_level") :
            bs = tf.shape(gt_boxes)[0]
            gt_boxes = tf.stack([gt_boxes[..., 1],
                                 gt_boxes[..., 0],
                                 gt_boxes[..., 3],
                                 gt_boxes[..., 2]], -1)
            targets = tf.concat([gt_boxes, tf.cast(gt_labels[:, :, None], gt_boxes.dtype)], -1)  # [b, n, 5] => [x1, y1, x2, y2, l]
            batch_inds = tf.tile(tf.reshape(tf.range(bs), [bs, 1, 1]), [1, tf.shape(gt_boxes)[1], 1]) 
            targets = tf.concat([tf.cast(batch_inds, targets.dtype), targets], -1)
            
            targets = tf.reshape(targets, [-1, 6])  # [b * n, 6] => [bi, x1, y1, x2, y2, l]
            targets = tf.boolean_mask(targets, targets[:, -1] > 0)
            anchor_inds = tf.tile(
                tf.reshape(tf.range(self.num_anchors), [self.num_anchors, 1, 1]), 
                [1, tf.shape(targets)[0], 1])  # [na, b * n, 1]
            targets = tf.tile(tf.expand_dims(targets, 0), [self.num_anchors, 1, 1])
            targets = tf.concat([targets, tf.cast(anchor_inds, targets.dtype)], -1)  # [na, b * n, 7] => [bi, x1, y1, x2, y2, l, ai]

            if self.cfg.label_assignment == "iou": 
                iou_ = self._calc_iou(targets[..., 1:4], anchors)
                mask = iou_ < self.cfg.anchor_threshold
                targets = tf.boolean_mask(targets, mask)
            else:
                tgt_wh = targets[..., 3:5] - targets[..., 1:3]   # [na, b * n, 2]
                ratios = tgt_wh / anchors[:, None]  # [na, b * n, 2]
                mask_ratios = tf.reduce_max(tf.maximum(ratios, 1. / ratios), -1) < self.cfg.anchor_threshold # [na, b * n]
                targets = tf.boolean_mask(targets, mask_ratios)  # [m1, 7]

                tgt_xy = (targets[..., 1:3] + targets[..., 3:5]) * 0.5  # [m1, 2]
                tgt_xy /= strides
                tgt_xy_inv = tf.cast(grid_shape, tgt_xy.dtype) - tgt_xy

                mask_xy = tf.transpose(tf.logical_and(tgt_xy % 1. < self.g, tgt_xy > 1.), [1, 0])  # [2, m1]
                mask_xy_inv = tf.transpose(tf.logical_and(tgt_xy_inv % 1. < self.g, tgt_xy_inv > 1.), [1, 0])  # [2, m1]
                mask = tf.concat([tf.cast(tf.ones([1, tf.shape(mask_xy)[1]]), tf.bool), mask_xy, mask_xy_inv], 0)  # [5, m1]
                targets = tf.boolean_mask(tf.tile(targets[None], [5, 1, 1]), mask)  # [m2, 7] => [bi, x1, y1, x2, y2, l, ai]
            
            outputs = tf.zeros([bs, grid_shape[0], grid_shape[1], self.num_anchors, 5], tf.float32)
            if tf.shape(targets)[0] <= 0:
                return outputs

            bi = tf.cast(targets[:, 0], tf.int32)
            ai = tf.cast(targets[:, -1], tf.int32)
            tgt_xy = (targets[:, 3:5] + targets[:, 1:3]) * 0.5
            if self.cfg.label_assignment != "iou":
                offsets = tf.zeros_like(tgt_xy)[None] + self.off[:, None]
                offsets = tf.boolean_mask(offsets, mask)
                gij = tf.cast(tgt_xy - offsets, tf.int32)
            else:
                gij = tf.cast(tgt_xy, tf.int32)
            indices = tf.stack([bi, gij[:, 1], gij[:, 0], ai], -1)
            
            outputs = tf.tensor_scatter_nd_update(outputs, indices, targets[..., 1:6])
            
            return outputs

    def compute_losses_per_level(self, predictions, targets, anchors, weights, strides):
        with tf.name_scope("compute_losses_per_level"):
            # bs = tf.cast(tf.shape(predictions)[0], tf.float32)
            predictions = tf.cast(predictions, tf.float32)            
            bs = tf.shape(predictions)[0]
            nx, ny = tf.shape(predictions)[2], tf.shape(predictions)[1]
            grid_xy = self.make_grid(nx, ny)
            grid_xy = tf.cast(grid_xy, predictions.dtype)
            grid_xy = tf.tile(grid_xy, [bs, 1, 1, 1, 1])

            anchors = tf.reshape(anchors, shape=[1, 1, 1, self.num_anchors, 2])
            anchors = tf.tile(anchors, [bs, ny, nx, 1, 1])
            pred_xy = (tf.nn.sigmoid(predictions[:, :, :, :, 0:2]) * 2. - 0.5 + grid_xy) * strides
            pred_hw = (tf.nn.sigmoid(predictions[:, :, :, :, 2:4]) * 2) ** 2 * anchors
            # pred_hw = tf.math.exp(predictions[:, :, :, :, 2:4]) * anchors
            pred_boxes = tf.concat([pred_xy - pred_hw * 0.5, pred_xy + pred_hw * 0.5], -1)
            # pred_xy = tf.nn.sigmoid(predictions[..., 0:2]) * 2. - 0.5
            # pred_wh = (tf.nn.sigmoid(predictions[..., 2:4]) * 2.) ** 2. * anchors
            # pred_boxes = tf.concat([pred_xy - pred_wh * 0.5, pred_xy + pred_wh * 0.5], -1)
            
            tgt_boxes = targets[..., 0:4]
            tgt_conf = tf.cast(targets[..., 4:5] > 0, tf.float32)
            
            num_pos = tf.reduce_sum(tgt_conf) + 1e-3
            iou = compute_iou(tgt_boxes, pred_boxes, "ciou")
            
            bbox_loss = tf.reduce_sum((1. - iou) * tf.squeeze(tgt_conf, -1)) / num_pos
            pred_labels = predictions[..., 5:]
            tgt_labels = tf.one_hot(tf.cast(targets[..., 4] - 1., tf.int32), self.num_classes)
            label_loss = self.label_loss_func(tgt_labels, pred_labels, tgt_conf)
            label_loss = tf.reduce_sum(label_loss) / (num_pos * self.num_classes)
            
            pred_conf = predictions[..., 4:5]
            tgt_conf *= ((1.0 - self.cfg.gr) + self.cfg.gr * iou[..., None])
            tgt_conf = tf.stop_gradient(tf.clip_by_value(tgt_conf, 0, 1))
            conf_loss = self.conf_loss_func(tgt_conf, pred_conf)
            conf_loss = tf.reduce_mean(conf_loss)
            
            bs = tf.cast(bs, tf.float32)
            return bbox_loss * bs, label_loss * bs, conf_loss * bs

    def compute_losses(self, predictions, image_info):
        with tf.name_scope("compute_losses"):
            bbox_loss_list = []
            conf_loss_list = []
            label_loss_list = []
            
            total_anchors = tf.convert_to_tensor(self.cfg.anchors, tf.float32)
            for i, level in enumerate(range(self.min_level, self.max_level + 1)):
                grid_shape = tf.shape(predictions[i])[1:3]
                strides = 2 ** level
                anchors = total_anchors[i]
                anchors = tf.reshape(anchors, [self.num_anchors, 2])
                
                targets = self.get_targets(
                    gt_boxes=tf.cast(image_info["boxes"], tf.float32),
                    gt_labels=tf.cast(image_info["labels"], tf.float32),
                    grid_shape=grid_shape, 
                    anchors=anchors,
                    strides=strides)
                targets = tf.stop_gradient(targets)
                
                bbox_loss, label_loss, conf_loss = self.compute_losses_per_level(
                    predictions[i], targets, anchors, self.cfg.balance[i], strides)

                bbox_loss_list.append(bbox_loss)
                conf_loss_list.append(conf_loss)
                label_loss_list.append(label_loss)

            bbox_loss = tf.add_n(bbox_loss_list)
            conf_loss = tf.add_n(conf_loss_list)
            label_loss = tf.add_n(label_loss_list)

            bbox_loss = bbox_loss * self.cfg.box_weight
            label_loss = label_loss * self.cfg.label_weight 
            conf_loss = conf_loss * self.cfg.conf_weight         

            return dict(bbox_loss=bbox_loss, conf_loss=conf_loss, label_loss=label_loss) 
         
    def make_grid(self, width, height):
        with tf.name_scope("make_grid"):
            grid_x, grid_y = tf.meshgrid(tf.range(width), tf.range(height)) 
            grid_xy =tf.stack([grid_x, grid_y], -1)  # (h, w, 2)
            grid_xy = tf.expand_dims(tf.expand_dims(grid_xy, 0), 3)  # (1, h, w, 1, 2)
        
            return grid_xy
        
    def get_boxes_v5(self, predictions):
        with  tf.name_scope("get_boxes_v5"):
            box_list = []
            conf_list = []
            score_list = []
            for i, _ in enumerate(range(self.min_level, self.max_level + 1)):
                pred = predictions[i]
                pred_shape = tf.shape(pred)
                bs, ny, nx = pred_shape[0], pred_shape[1], pred_shape[2]
                pred = tf.reshape(pred, [bs, ny, nx, self.cfg.num_anchors, self.num_classes + 5])

                grid_xy = self.make_grid(nx, ny)
                grid_xy = tf.cast(grid_xy, pred.dtype)
                grid_xy = tf.tile(grid_xy, [bs, 1, 1, 1, 1])

                out = tf.sigmoid(pred)
                anchors = tf.constant(self.cfg.anchors[i], shape=[1, 1, 1, self.num_anchors, 2], dtype=out.dtype)
                anchors = tf.tile(anchors, [bs, ny, nx, 1, 1])
                strides = self.cfg.strides[i]
                
                out_xy = (out[..., 0:2] * 2. - 0.5 + grid_xy) * strides
                out_wh = (out[..., 2:4] * 2.) ** 2. * anchors
                
                out_boxes = tf.stack([out_xy[..., 1] - out_wh[..., 1] * 0.5, 
                                      out_xy[..., 0] - out_wh[..., 0] * 0.5,
                                      out_xy[..., 1] + out_wh[..., 1] * 0.5,
                                      out_xy[..., 0] + out_wh[..., 0] * 0.5], -1)
                
                boxes = tf.reshape(out_boxes, [bs, ny * nx * self.num_anchors, 4])
                conf = tf.reshape(out[..., 4], [bs, ny * nx * self.num_anchors, 1])
                score = tf.reshape(out[..., 5:], [bs, ny * nx * self.num_anchors, self.num_classes])
                input_size = tf.convert_to_tensor([[ny * strides, nx * strides]], boxes.dtype)
                input_size = tf.tile(input_size, [bs, 1])
                boxes = box_utils.to_normalized_coordinates(
                    boxes, input_size[:, 0:1, None], input_size[:, 1:2, None])
                
                box_list.append(boxes)
                conf_list.append(conf)
                score_list.append(score)
            
            pred_boxes = tf.cast(tf.concat(box_list, 1), tf.float32)
            pred_conf = tf.cast(tf.concat(conf_list, 1), tf.float32)
            pred_scores = tf.cast(tf.concat(score_list, 1), tf.float32)   
            
            if "Quality" in self.cfg.test.nms:
                return self.nms(pred_boxes, pred_scores, pred_conf)
            
            pred_scores = pred_scores * pred_conf
                
            return self.nms(pred_boxes, pred_scores)
    
    def get_boxes_v3(self, predictions):
        with tf.name_scope("get_boxes"):
            box_list = []
            conf_list = []
            score_list = []
            for i, _ in enumerate(range(self.cfg.min_level, self.cfg.max_level + 1)):
                pred = tf.cast(predictions[i], tf.float32)
                pred_shape = tf.shape(pred)
                bs, ny, nx = pred_shape[0], pred_shape[1], pred_shape[2]
                pred = tf.reshape(pred, [bs, ny, nx, self.cfg.num_anchors, 5 + self.cfg.num_classes])

                grid_xy = tf.cast(self.make_grid(nx, ny), dtype=pred.dtype)
                grid_xy = tf.tile(grid_xy, [bs, 1, 1, 1, 1])
                strides = self.cfg.strides[i]
                box_xy = (tf.nn.sigmoid(pred[..., 0:2]) + grid_xy) * strides
               
                anchors = tf.constant(self.cfg.anchors[i], shape=[1, 1, 1, self.cfg.num_anchors, 2], dtype=pred.dtype)
                anchors = tf.tile(anchors, [bs, 1, 1, 1, 1])
                box_wh = tf.math.exp(pred[..., 2:4]) * anchors
                boxes = tf.stack([box_xy[..., 1] - box_wh[..., 1] * 0.5,
                                  box_xy[..., 0] - box_wh[..., 0] * 0.5,
                                  box_xy[..., 1] + box_wh[..., 1] * 0.5,
                                  box_xy[..., 0] + box_wh[..., 0] * 0.5], -1)

                conf = tf.nn.sigmoid(pred[..., 4])
                score = tf.nn.sigmoid(pred[..., 5:])

                boxes = tf.reshape(boxes, [bs, -1, 4])
                conf = tf.reshape(conf, [bs, -1, 1])
                score = tf.reshape(score, [bs, -1, self.cfg.num_classes])

                input_size = tf.convert_to_tensor([[ny * strides, nx * strides]], boxes.dtype)
                input_size = tf.tile(input_size, [bs, 1])
                boxes = box_utils.to_normalized_coordinates(
                    boxes, input_size[:, 0:1, None], input_size[:, 1:2, None])

                box_list.append(boxes)
                conf_list.append(conf)
                score_list.append(score)

            pred_boxes = tf.concat(box_list, 1)
            pred_conf = tf.concat(conf_list, 1)
            pred_scores = tf.concat(score_list, 1)

            if "Quality" in self.cfg.test.nms:
                return self.nms(pred_boxes, pred_scores, pred_conf)
            
            pred_scores = pred_scores * pred_conf
                
            return self.nms(pred_boxes, pred_scores)

    def get_boxes(self, predictions):
        if self.cfg.label_assignment == "iou":
            return self.get_boxes_v3(predictions)

        return self.get_boxes_v5(predictions)
    
    @tf.function
    def __call__(self, inputs, training=None):
        return self.detector(inputs, training=None)

    def save_weights(self, name):
        self.detector.save_weights(name)


def _load_darknet_weights(model, darknet_weights_path, num_convs=91):
    wf = open(darknet_weights_path, "rb")
    major, minor, revision, seen, _ = np.fromfile(wf, np.int32, 5)

    for i in range(num_convs):
        layer = model.model.get_layer("conv" + str(i+1))
        if isinstance(layer, ConvNormActBlock):
            kernel = layer.conv.kernel
            # print("conv" + str(i + 1), kernel.shape.as_list())  
            gamma = layer.norm.gamma
            beta = layer.norm.beta
            moving_mean = layer.norm.moving_mean
            moving_variance = layer.norm.moving_variance
            
            ksize, _, infilters, filters = kernel.shape.as_list()
            dshape = (filters, infilters, ksize, ksize)

            beta.assign(np.fromfile(wf, np.float32, filters))
            gamma.assign(np.fromfile(wf, np.float32, filters))
            moving_mean.assign(np.fromfile(wf, np.float32, filters))
            moving_variance.assign(np.fromfile(wf, np.float32, filters))

            dkernel = np.fromfile(wf, np.float32, np.product(dshape))
            dkernel = dkernel.reshape(dshape).transpose([2, 3, 1, 0])
            kernel.assign(dkernel)
        
        if isinstance(layer, tf.keras.layers.Conv2D):
            kernel = layer.kernel
            # print("conv" + str(i + 1), kernel.shape.as_list())  
            bias = layer.bias
            ksize, _, infilters, filters = kernel.shape.as_list()
            dshape = (filters, infilters, ksize, ksize)
            bias.assign(np.fromfile(wf, np.float32, filters))
            dkernel = np.fromfile(wf, np.float32, np.product(dshape))
            dkernel = dkernel.reshape(dshape).transpose([2, 3, 1, 0])
            kernel.assign(dkernel)

    assert len(wf.read()) == 0, "Failed to read all data"
    wf.close()


def _load_weight_from_torch(model, torch_weights):
    import torch
    
    torch_model = torch.load(torch_weights, map_location=torch.device("cpu"))
    # for k, v in torch_model["state_dict"].items():
    #     if "tracked" in k:
    #         continuel
    #     print(k, v.shape)
        
    for weight in model.weights:
        name = weight.name
        name = name.split(":")[0]
        name = name.replace("/", ".")
        if "batch_normalization" in name:
            name = name.replace("batch_normalization", "bn")
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
        if "yolo_v5" in name:
            name = name.replace("yolo_v5.", "")
        if "conv2d.weight" in name:
            name = name.replace("conv2d.weight", "conv.weight")
        if "predicted" in name:
            name = name.replace("predicted", "m")
        name = "model." + name
        # print(name, weight.shape)

        tw = torch_model["state_dict"][name].numpy()
        if len(tw.shape) == 4:
            tw = np.transpose(tw, (2, 3, 1, 0))
        
        if len(tw.shape) == 2:
            tw = np.transpose(tw, (1, 0))
        weight.assign(tw)


def fuse_conv_and_bn(conv, bn):
    fused_conv = tf.keras.layers.Conv2D(filters=conv.filters,
                                        kernel_size=conv.kernel_size,
                                        strides=conv.strides,
                                        padding=conv.padding,
                                        dilation_rate=conv.dilation_rate,
                                        use_bias=True)
    kernel_shape = tf.shape(conv.kernel)
    fused_conv(tf.random.uniform([1, 32, 32, kernel_shape[-2]]))
    
    # prepare kernel
    # conv_kernel = tf.transpose(tf.reshape(conv.kernel, [-1, kernel_shape[0]]))
    # bn_kernel = tf.linalg.diag(bn.gamma / tf.sqrt(bn.epsilon + bn.moving_variance))
    
    # kernel = tf.matmul(bn_kernel, conv_kernel)
    # kernel = tf.reshape(tf.transpose(conv_kernel), kernel_shape)
    bn_kernel = bn.gamma / tf.sqrt(bn.epsilon + bn.moving_variance)
    kernel = conv.kernel * bn_kernel
    fused_conv.kernel.assign(kernel)

    # prepare bias
    conv_bias = tf.zeros([kernel_shape[-1]], conv.kernel.dtype) if conv.bias is None else conv.bias
    bn_bias = bn.beta - bn.gamma * bn.moving_mean / tf.sqrt(bn.epsilon + bn.moving_variance)
    # bias = tf.reshape(tf.matmul(bn_kernel, tf.reshape(conv_bias, [-1, 1])), [kernel_shape[-1]]) + bn_bias
    bias = bn_kernel * conv_bias + bn_bias
    fused_conv.bias.assign(bias)

    return fused_conv


def fuse(model):
    print("Fusing layers ...")
    for l in model.layers:
        if isinstance(l, ConvNormActBlock) and isinstance(l.norm, tf.keras.layers.BatchNormalization):
            l.conv = fuse_conv_and_bn(l.conv, l.norm)
            l.norm = None
            l.call = l.fused_call
            # print("Fused ", l.name)


if __name__ == "__main__":
    import cv2
    import numpy as np
    from demo import coco_id_mapping, draw, random_color
    from configs.yolov4_config import get_yolov4_config
    from data.datasets.coco_dataset import COCODataset
    from core.metrics.mean_average_precision import mAP

    np.set_printoptions(precision=2)

    num_layers = {
        "YOLOv4": 110,
        "YOLOv4-tiny": 21,
        "YOLOv4-csp": 115,
        "YOLOv4x-mish": 137,
        "ScaledYOLOv4": 0,
    }

    label_assignments = {
        "YOLOv4": "iou",
        "YOLOv4-tiny": "iou",
        "YOLOv4-csp": "wh",
        "YOLOv4x-mish": "wh",
        "ScaledYOLOv4-p5": "wh",
        "ScaledYOLOv4-p6": "wh",
        "ScaledYOLOv4-p7": "wh",
    }

    name = "ScaledYOLOv4-p7"
    cfg = get_yolov4_config(name=name,
                            # max_level=max_level,
                            # min_level=min_level,
                            # strides=strides,
                            # anchors=anchors,
                            # num_anchors=3,
                            label_assignment=label_assignments[name])

    def _assingment_test():
        image = None

    def _convert():
        model = YOLOv4(cfg, training=False)
        
        # _load_darknet_weights(model, "/home/bail/Downloads/%s.weights" % name.lower(), num_layers[name])
        _load_weight_from_torch(model.detector, "/home/bail/Downloads/ScaledYOLOv4-yolov4-large/yolov4-p7.pth")
        
        # fuse(model.model)
        # with tf.io.gfile.GFile("/home/bail/Workspace/TRTNets/images/bus.jpg", "rb") as gf:
        #     img = tf.image.decode_image(gf.read())
        
        # img = tf.image.resize(img, cfg.input_size)
        img = cv2.imread("/home/bail/Workspace/TRTNets/images/bus.jpg")
        img = cv2.resize(img, cfg.input_size)
        inp = img[:, :, ::-1]
        inp = tf.convert_to_tensor(inp[None], tf.float32)

        # @tf.function
        # def _fn(x):
        #     outputs = model.detector(inp, training=False)
        #     return model.get_boxes(outputs)
        
        outs = model(inp)

        num = outs["valid_detections"].numpy()[0]
        boxes = outs["nmsed_boxes"].numpy()[0]
        scores = outs["nmsed_scores"].numpy()[0]
        classes = outs["nmsed_classes"].numpy()[0]
       
        for i in range(num):
            box = boxes[i] * cfg.input_size[0]
            # box = boxes[i] * np.array([height, width, height, width])
            c = classes[i] + 1
            print(box, c)
            img = draw(img, box, c, scores[i], coco_id_mapping, random_color(int(c)))
        
        cv2.imshow("img", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
        tf.saved_model.save(model.detector, "./%s" % name)
        model.save_weights("%s.h5" % name)

    _convert()
