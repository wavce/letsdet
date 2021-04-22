import numpy as np
import tensorflow as tf 
from ..builder import ANCHOR_GENERATORS


@ANCHOR_GENERATORS.register
class AnchorGeneratorV2(tf.keras.layers.Layer):
    """Standard anchor generator for 2D anchor-based detectors.

    Args:
        strides (int): Strides of anchors in current feature levels.
        aspect_ratios (list[float]): The list of ratios between the height and width
            of anchors in a current level.
        scales (list[int] | None): Anchor scales for anchors in a current level.
            It cannot be set at the same time if `octave_base_scale` and
            `scales_per_octave` are set.
        base_sizes (list[int] | None): The basic sizes
            of anchors in multiple levels.
            If None is given, strides will be used as base_sizes.
            (If strides are non square, the shortest stride is taken.)
        scale_major (bool): Whether to multiply scales first when generating
            base anchors. If true, the anchors in the same row will have the
            same scales.
        octave_base_scale (int): The base scale of octave.
        scales_per_octave (int): Number of scales for each octave.
            `octave_base_scale` and `scales_per_octave` are usually used in
            retinanet and the `scales` should be None when they are set.
        centers (list[float, float] | None): The centers of the anchor
            relative to the feature grid center in single feature levels.
            By default it is set to be None and not used. If a list of 
            float is given, they will be used to shift the centers of anchors.
        center_offset (float): The offset of center in propotion to anchors'
            width and height. 
    """

    def __init__(self,
                 strides,
                 aspect_ratios,
                 scales=None,
                 base_size=None,
                 scale_major=True,
                 octave_base_scale=None,
                 scales_per_octave=None,
                 center=None,
                 center_offset=0.,
                 num_anchors=None,
                 **kwargs):
        super(AnchorGeneratorV2, self).__init__(**kwargs)
       
        # calculate base size of anchors
        self.strides = strides
        self.base_size = strides if base_size is None else base_size
        
        # calculate scales of anchors
        assert ((octave_base_scale is not None
                and scales_per_octave is not None) ^ (scales is not None)), \
            'scales and octave_base_scale with scales_per_octave cannot' \
            ' be set at the same time'
        if scales is not None:
            self.scales = scales
        elif octave_base_scale is not None and scales_per_octave is not None:
            octave_scales = np.array(
                [2**(i / scales_per_octave) for i in range(scales_per_octave)])
            scales = octave_scales * octave_base_scale
            self.scales = scales
        else:
            raise ValueError('Either scales or octave_base_scale with '
                             'scales_per_octave should be set')

        self.octave_base_scale = octave_base_scale
        self.scales_per_octave = scales_per_octave
        self.ratios = aspect_ratios
        self.scale_major = scale_major
        self.center = center
        self.center_offset = center_offset
    
    def build(self, input_shape):
        self.ratios = tf.convert_to_tensor(self.ratios, dtype=tf.float32)
        self.scales = tf.convert_to_tensor(self.scales, dtype=tf.float32)

    def gen_base_anchors(self):
        """Generate base anchors of a single level.
        """
        w = self.base_size
        h = self.base_size
      
        if self.center is None:
            x_center = self.center_offset * w
            y_center = self.center_offset * h
        else:
            x_center, y_center = self.center

        h_ratios = tf.math.sqrt(self.ratios)
        w_ratios = 1 / h_ratios
        if self.scale_major:
            ws = tf.reshape(w * w_ratios[:, None] * self.scales[None, :], [-1])
            hs = tf.reshape(h * h_ratios[:, None] * self.scales[None, :], [-1])
        else:
            ws = tf.reshape(w * self.scales[:, None] * w_ratios[None, :], [-1])
            hs = tf.reshape(h * self.scales[:, None] * h_ratios[None, :], [-1])

        # use float anchor and the anchor's center is aligned with the
        # pixel center
        base_anchors = [
            x_center - 0.5 * ws, y_center - 0.5 * hs, 
            x_center + 0.5 * ws, y_center + 0.5 * hs
        ]
        base_anchors = tf.stack(base_anchors, axis=-1)

        return base_anchors

    def call(self, feat_height, feat_width):
        base_anchors = self.gen_base_anchors()
        
        xx, yy = tf.meshgrid(tf.range(0, feat_width * self.strides, self.strides),
                             tf.range(0, feat_height * self.strides, self.strides))
        xx = tf.reshape(xx, [-1])
        yy = tf.reshape(yy, [-1])
        shifts = tf.stack([xx, yy, xx, yy], axis=-1)
        shifts = tf.cast(shifts, base_anchors.dtype)
        # first feat_w elements correspond to the first row of shifts
        # add A anchors (1, A, 4) to K shifts (K, 1, 4) to get
        # shifted anchors (K, A, 4), reshape to (K*A, 4)

        all_anchors = base_anchors[None, :, :] + shifts[:, None, :]
        all_anchors = tf.reshape(all_anchors, [-1, 4])
   
        # first A rows correspond to A anchors of (0, 0) in feature map,
        # then (0, 1), (0, 2), ...
        return all_anchors
    
    def get_config(self):
        layer_config = {
            "strides": self.strides,
            "aspect_ratios": self.ratios.numpy(),
            "scales": self.scales.numpy(),
            "base_size": self.base_size,
            "octave_base_scale": self.octave_base_scale,
            "scales_per_octave": self.scales_per_octave,
            "scale_major": self.scale_major,
            "center": self.center,
            "center_offset": self.center_offset
        }

        layer_config.update(super(AnchorGeneratorV2, self).get_config())
        return layer_config


if __name__ == "__main__":
    generator = AnchorGeneratorV2(octave_base_scale=8,
                                  scales_per_octave=1, 
                                  aspect_ratios=[1.], 
                                  strides=128)
    print(generator(8, 8))

