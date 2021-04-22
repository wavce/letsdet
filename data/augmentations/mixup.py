import tensorflow as tf
import tensorflow_probability as tfp
from ..builder import AUGMENTATIONS 


@AUGMENTATIONS.register
class Mixup(object):
    def __init__(self, batch_size, alpha, prob=0.5, max_boxes=200):
        self.alpha = alpha
        self.batch_size = batch_size
        self.prob = prob
        self.max_boxes = max_boxes
    
    def _mixup(self, images, boxes, labels):
        """Applies Mixup regularization to a batch of images and labels.

        [1] Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, David Lopez-Paz
            Mixup: Beyond Empirical Risk Minimization.
            ICLR'18, https://arxiv.org/abs/1710.09412

        Args:
            images: A batch of images of shape [batch_size, ...]
            labels: A batch of labels of shape [batch_size, num_classes]

        Returns:
            A tuple of (images, boxes, labels) with the same dimensions as the input with
            Mixup regularization applied.
        """
        mix_weight = tfp.distributions.Beta(self.alpha, self.alpha).sample([self.batch_size, 1])
        mix_weight = tf.maximum(mix_weight, 1. - mix_weight)
        images_mix_weight = tf.reshape(mix_weight, [self.batch_size, 1, 1, 1])
        # Mixup on a single batch is implemented by taking a weighted sum with the same batch in reverse.
        image_dtype = images.dtype
        images = tf.cast(images, mix_weight.dtype)
        images_mix = images * images_mix_weight + images[::-1] * (1. - images_mix_weight)
        
        boxes_mix = tf.concat([boxes, boxes[::-1]], 1)
        labels_mix = tf.concat([labels, labels[::-1]], 1)

        def _fn(b, l):
            valid = l != 0
            l = tf.boolean_mask(l, valid)
            b = tf.boolean_mask(b, valid)
            num = tf.size(l)
            if num < self.max_boxes:
                l = tf.concat([l, tf.zeros([self.max_boxes - num], l.dtype)], 0)
                b = tf.concat([b, tf.zeros([self.max_boxes - num, 4], b.dtype)], 0)
            else:
                l = l[:self.max_boxes]
                b = b[:self.max_boxes]
            
            return b, l
        
        boxes_mix, labels_mix = tf.map_fn(
            lambda inp: _fn(*inp), 
            elems=(boxes_mix, labels_mix),
            fn_output_signature=(boxes_mix.dtype, labels_mix.dtype))

        images_mix = tf.cast(images_mix, image_dtype)

        return images_mix, boxes_mix, labels_mix
    
    def __call__(self, images, images_info):
        with tf.name_scope("mixup"):
            images = tf.cast(images, tf.uint8)
            images, images_info["boxes"], images_info["labels"] = tf.cond(
                tf.random.uniform([]) >= self.prob,
                lambda: self._mixup(images, images_info["boxes"], images_info["labels"]),
                lambda: (images, images_info["boxes"], images_info["labels"]))
            
            return images, images_info
