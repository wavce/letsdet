from .bbox_transform import Box2Delta
from .bbox_transform import Delta2Box
from .overlaps import compute_iou 
from .overlaps import compute_unaligned_iou
from .bbox_transform import Distance2Box
from .bbox_transform import Box2Distance


def build_decoder(decoder, **kwargs):
    if decoder == "Delta2Box":
        return Delta2Box(**kwargs)
    
    if decoder == "Distance2Box":
        return Distance2Box()
    
    raise TypeError("Could not interpret bbox decoder function identifier: {}".format(repr(decoder)))


def build_encoder(encoder, **kwargs):
    if encoder == "Box2Delta":
        return Box2Delta(**kwargs)
    
    if encoder == "Box2Distance":
        return Box2Distance()
    
    raise TypeError("Could not interpret bbox encoder function identifier: {}".format(repr(encoder)))


__all__ = [
    "build_encoder",
    "build_decoder",
    "compute_iou",
    "compute_unaligned_iou"
]
