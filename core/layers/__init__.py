import tensorflow as tf
import tensorflow_addons as tfa
from .activations import Mish
from .scale import Scale
from .max_in_out import MaxInOut
from .drop_block import DropBlock2D
from .nms import FastNonMaxSuppression
from .nms import NonMaxSuppression
from .nms import CombinedNonMaxSuppression
from .nms import SoftNonMaxSuppression
# from .normalizations import L2Normalization
from .dcnv2 import DCNv2
from .normalizations import GroupNormalization
from .nearest_upsamling import NearestUpsampling2D
from .weight_standardization_conv2d import WSConv2D
from .normalizations import FilterResponseNormalization
from .normalizations import InstanceNormalization
from .normalizations import FrozenBatchNormalization
from .proposal_layer import ProposalLayer
from .position_sensitive_roi_pooling import PSRoIPooling
from .position_sensitive_average_pooling import PSAvgPooling
from .roi_pooling import SingleLevelAlignedRoIPooling, MultiLevelAlignedRoIPooling


def build_convolution(convolution, **kwargs):
    if convolution == "depthwise_conv2d":
        return tf.keras.layers.DepthwiseConv2D(**kwargs)
    elif convolution == "wsconv2d":
        return WSConv2D(**kwargs)
    elif convolution == "conv2d":
        return tf.keras.layers.Conv2D(**kwargs)
    elif convolution == "separable_conv2d":
        return tf.keras.layers.SeparableConv2D(**kwargs)
    elif convolution == "dcnv2":
        return DCNv2(**kwargs)
    else:
        raise TypeError("Could not interpret convolution function identifier: {}".format(repr(convolution)))


def build_normalization(normalization, **kwargs):
    if normalization == "group_norm":
        return GroupNormalization(**kwargs)
    elif normalization == "batch_norm":
        return tf.keras.layers.BatchNormalization(**kwargs)
    elif normalization == "frozen_batch_norm":
        return FrozenBatchNormalization(**kwargs)
    # elif normalization == "switchable_norm":
    #     return SwitchableNormalization(**kwargs)
    elif normalization == "filter_response_norm":
        return FilterResponseNormalization(**kwargs)
    elif normalization == "sync_batch_norm":
        return tf.keras.layers.experimental.SyncBatchNormalization(**kwargs)
    else:
        raise TypeError("Could not interpret normalization function identifier: {}".format(
            repr(normalization)))


def build_activation(**kwargs):
    if kwargs["activation"] == "leaky_relu":
        kwargs.pop("activation")
        return tf.keras.layers.LeakyReLU(**kwargs)
    if kwargs["activation"] == "mish":
        kwargs.pop("activation")
        return Mish(**kwargs)

    return tf.keras.layers.Activation(**kwargs)


def build_roi_pooling(roi_pooling, **kwargs):
    if roi_pooling == "SingleLevelAlignedRoIPooling":
        return SingleLevelAlignedRoIPooling(**kwargs)
    
    if roi_pooling == "MultiLevelAlignedRoIPooling":
        return MultiLevelAlignedRoIPooling(**kwargs)

    if roi_pooling == "PSRoIPooling":
        return PSRoIPooling(**kwargs)
    
    if roi_pooling == "PSAvgPooling":
        return PSAvgPooling(**kwargs)
    
    raise TypeError("Could not interpret roi_pooling function identifier: {}".format(repr(roi_pooling)))


__all__ = [
    "Scale",
    "MaxInOut",
    "DropBlock2D",
    "L2Normalization",
    "build_activation",
    "build_convolution",
    "build_normalization",
    "NearestUpsampling2D",
    "build_roi_pooling",
    "ProposalLayer"
]
