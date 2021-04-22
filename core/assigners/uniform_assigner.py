import tensorflow as tf 
from .assigner import Assigner
from ..builder import ASSIGNERS 
from utils import box_utils


@ASSIGNERS.register
class UniformAssigner(Assigner):
    """
        Uniform Matching between the anchors and gt boxes, which can achieve
        balance in positive anchors.

        Args:
            match_times(int): Number of positive anchors for each gt box.
    """
    def __init__(self, 
                 pos_ignore_thresh: float = 0.7,
                 neg_ignore_thresh: float = 0.15, 
                 match_times: int = 4, **kwargs):
        super(UniformAssigner, self).__init__(**kwargs)

        self.match_times = match_times
        self.pos_ignore_thresh = pos_ignore_thresh
        self.neg_ignore_thresh = neg_ignore_thresh
    
    def _cdist(self, x, y):
        with tf.name_scope("cdist"):
            x = tf.expand_dims(x, 0)
            y = tf.expand_dims(y, 1)

            dist = tf.sqrt(tf.reduce_sum(tf.square(x - y), -1))

            return dist
    
    def assign(self, gt_boxes, gt_labels, anchors, predicted_boxes):
        with tf.name_scope("assign"):
            # Compute the L1 cost between boxes
            # Note that we use anchors and predict boxes both
            C = self._cdist(predicted_boxes, gt_boxes)
            C1 = self._cdist(anchors, gt_boxes)

            _, indices = tf.nn.top_k(C, k=self.match_times)
            _, indices2 = tf.nn.top_k(C1, k=self.match_times)

            indices = tf.transpose(indices)
            indices2 = tf.transpose(indices2)
            indices = tf.reshape(indices, [-1, 1])
            indices2 = tf.reshape(indices2, [-1, 1])
            indices = tf.concat([indices, indices2], 0)
            gt_boxes = tf.tile(gt_boxes, [self.match_times * 2, 1])
            gt_labels = tf.tile(gt_labels, [self.match_times * 2])

            anchor_ious = box_utils.bbox_overlap(anchors, gt_boxes)
            pos_anchor_ious = tf.gather_nd(anchor_ious, tf.concat([indices, gt_labels[:, None]], -1))
            pos_ignore_mask = pos_anchor_ious < self.neg_ignore_thresh

            gt_labels = tf.where(pos_ignore_mask, 0 - tf.ones_like(gt_labels), gt_labels)
            
            tgt_boxes = tf.scatter_nd(indices, gt_boxes, tf.shape(predicted_boxes))
            tgt_labels = tf.scatter_nd(indices, gt_labels, tf.shape(predicted_boxes[:, 0]))

            pred_ious = box_utils.bbox_overlap(predicted_boxes, gt_boxes) 
            pred_max_ious = tf.reduce_max(pred_ious, 1)

            neg_ignore_mask = pred_max_ious > self.neg_ignore_thresh
            tgt_labels = tf.where(neg_ignore_mask, 0 - tf.ones_like(tgt_labels), tgt_labels)

            return tgt_boxes, tgt_labels

    def __call__(self, gt_boxes, gt_labels, anchors, pred_boxes):
        return self.assign(gt_boxes, gt_labels, anchors, pred_boxes)



if __name__ == "__main__":
    pboxes = tf.random.uniform([100, 4])
    anchors = tf.random.uniform([100, 4])

    gt_boxes = tf.random.uniform([2, 4])
    gt_labels = tf.constant([2, 3])

    assigner = UniformAssigner(8)
    assigner.assign(gt_boxes, gt_labels, anchors, pboxes)