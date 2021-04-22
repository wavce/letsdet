import math
import tensorflow as tf
from ..builder import ANCHOR_GENERATORS


@ANCHOR_GENERATORS.register
class AnchorGenerator(tf.keras.layers.Layer):
    def __init__(self, scales, aspect_ratios, strides, offset=0, shifting=False, num_anchors=None, **kwargs):
        super(AnchorGenerator, self).__init__(**kwargs)

        self.shifting = shifting
        self.scales = scales
        self.aspect_ratios = aspect_ratios
        self.strides = strides
        self.offset = offset
        self.num_anchors = len(self.scales) * len(self.aspect_ratios) if num_anchors is None else num_anchors
    
    def build(self, input_shape):
        self.scales = tf.convert_to_tensor(self.scales, dtype=self.dtype)
        self.aspect_ratios = tf.convert_to_tensor(self.aspect_ratios, dtype=self.dtype)
        
        super(AnchorGenerator, self).build(input_shape)

    def call(self, feat_height, feat_width):
        start = self.strides * self.offset
        xx, yy = tf.meshgrid(tf.range(start, feat_width * self.strides, self.strides),
                             tf.range(start, feat_height * self.strides, self.strides))

        yy = tf.cast(yy, self.dtype)
        xx = tf.cast(xx, self.dtype)

        h_ratio = tf.math.sqrt(self.aspect_ratios)
        w_ratio = 1. / h_ratio
        ws = tf.reshape(self.scales[:, None] * w_ratio[None, :], [-1])
        hs = tf.reshape(self.scales[:, None] * h_ratio[None, :], [-1])
      
        yy = tf.expand_dims(yy, -1)
        xx = tf.expand_dims(xx, -1)
        anchors = tf.stack([xx - 0.5 * ws, yy - 0.5 * hs, xx + 0.5 * ws, yy + 0.5 * hs], axis=-1)

        anchors = tf.reshape(anchors, [feat_height * feat_width * self.num_anchors, 4])

        return anchors
    
    def get_config(self):
        layer_config = {
            "strides": self.strides,
            "aspect_ratios": self.aspect_ratios.numpy(),
            "scales": self.scales.numpy(),
            "num_anchors": self.num_anchors
        }

        layer_config.update(super(AnchorGenerator, self).get_config())
        
        return layer_config


def generate_base_anchors(sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.5, 1, 2)):
    """
    Generate base anchor boxes, which are continuous geometric rectangles
    centered on one feature map point sample. We can later build the set of anchors
    for the entire feature map by tiling these tensors; see `meth:grid_anchors`.
    Args:
        sizes (tuple[float]): Absolute size of the anchors in the units of the input
            image (the input received by the network, after undergoing necessary scaling).
            The absolute size is given as the side length of a box.
        aspect_ratios (tuple[float]]): Aspect ratios of the boxes computed as box
            height / width.
    Returns:
        Tensor of shape (len(sizes) * len(aspect_ratios), 4) storing anchor boxes
            in XYXY format.
    """

    # This is different from the anchor generator defined in the original Faster R-CNN
    # code or Detectron. They yield the same AP, however the old version defines cell
    # anchors in a less natural way with a shift relative to the feature grid and
    # quantization that results in slightly different sizes for different aspect ratios.
    # See also https://github.com/facebookresearch/Detectron/issues/227

    anchors = []
    for size in sizes:
        area = size ** 2.0
        for aspect_ratio in aspect_ratios:
            # s * s = w * h
            # a = h / w
            # ... some algebra ...
            # w = sqrt(s * s / a)
            # h = a * w
            w = math.sqrt(area / aspect_ratio)
            h = aspect_ratio * w
            x0, y0, x1, y1 = -w / 2.0, -h / 2.0, w / 2.0, h / 2.0
            anchors.append([x0, y0, x1, y1])
    return anchors



if __name__ == "__main__":
    import numpy as np
    import torch

    def _create_grid_offsets(size, stride, offset):
        grid_height, grid_width = size
        shifts_start = offset * stride
        shifts_x = torch.arange(
            shifts_start, grid_width * stride + shifts_start, 
            step=stride,
            dtype=torch.float32
        )
        shifts_y = torch.arange(
            shifts_start, grid_height * stride + shifts_start, 
            step=stride,
            dtype=torch.float32
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)

        return shift_x, shift_y

    generator = AnchorGenerator(scales=[32, 64, 128, 256, 512], 
                                aspect_ratios=[0.5, 1., 2.], 
                                strides=32)
    print(generator(8, 8))

    anchors = generate_base_anchors()
    anchors = torch.tensor(anchors)
    xx, yy = _create_grid_offsets((8, 8), 32, 0)
    shifts = torch.stack((xx, yy, xx, yy), dim=1)
    print((shifts.view(-1, 1, 4) + anchors.view(1, -1, 4)).reshape(-1, 4))
