import tensorflow as tf
from .sampler import Sampler
from ..builder import SAMPLERS


@SAMPLERS.register
class RandomSampler(Sampler):
    def __init__(self, num_proposals, pos_fraction, neg_pos_ub=-1, add_gt_as_proposals=True, **kwargs):
        super(RandomSampler, self).__init__(num_proposals, pos_fraction, neg_pos_ub, add_gt_as_proposals)
            
    def _random_choice(self, indices, num):
        return tf.random.shuffle(indices)[:num]

    def _sample_positive(self, assigned_labels, num_expected_proposals, **kwargs):
        pos_inds = tf.squeeze(tf.where(assigned_labels >= 1), 1)
        
        if tf.size(pos_inds) <= num_expected_proposals:
            return pos_inds
        
        return self._random_choice(pos_inds, num_expected_proposals)
    
    def _sample_negative(self, assigned_labels, num_expected_proposals, **kwargs):
        neg_inds = tf.squeeze(tf.where(assigned_labels == 0), 1)
        if tf.size(neg_inds) <= num_expected_proposals:
            return neg_inds
        
        return self._random_choice(neg_inds, num_expected_proposals)        
