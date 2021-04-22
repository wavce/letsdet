import tensorflow as tf
from .sampler import Sampler
from ..builder import SAMPLERS


@SAMPLERS.register
class PseudoSampler(Sampler):
    def __init__(self, **kwargs):
        pass
    
    def _sample_positive(self, assigned_labels, num_expected_proposals, **kwargs):
        raise NotImplementedError
    
    def _sample_negative(self, assigned_labels, num_expected_proposals, **kwargs):
        raise NotImplementedError

    def sample(self, assigned_boxes, assigned_labels, **kwargs):
        """Sample positive and negative boxes.

            Args:
                assigned_boxes (Tensor): The assigned boxes in assigner.
                assigned_labels (Tensor): The assigned labels in assigner.
            
            Returns:
                A dict -> target_boxes, target_labels, box_weights, label_weights
        """
        pos_mask = assigned_labels >= 1
        box_weights = tf.cast(pos_mask, tf.float32)
        
        valid_mask = assigned_labels >= 0
        target_labels = tf.where(valid_mask, tf.cast(assigned_labels, tf.int64), tf.zeros_like(assigned_labels, tf.int64))
        label_weights = tf.cast(valid_mask, tf.float32)

        return assigned_boxes, target_labels, box_weights, label_weights
        