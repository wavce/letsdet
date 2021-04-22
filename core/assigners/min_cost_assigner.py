import tensorflow as tf
from .assigner import Assigner
from ..builder import ASSIGNERS
from core.bbox import compute_unaligned_iou


@ASSIGNERS.register
class MinCostAssigner(Assigner):
    def __init__(self, class_weight=1., l1_weight=1., iou_weight=1., iou_type="giou", alpha=0.25, gamma=2., **kwargs):
        super(MinCostAssigner, self).__init__(**kwargs)

        self._class_weights = class_weight
        self._l1_weight = l1_weight
        self._iou_weight = iou_weight
        self._iou_type = iou_type

        self._gamma = gamma
        self._alpha = alpha
    
    def assign(self, gt_boxes, gt_labels, pred_boxes, pred_labels):
        with tf.name_scope("assign"):
            valid_mask = gt_labels > 0
            gt_labels = tf.boolean_mask(gt_labels, valid_mask) - 1
            gt_boxes = tf.boolean_mask(gt_boxes, valid_mask)
            
            # Compute the classification cost.
            num_classes = tf.shape(pred_labels)[-1]
            hw = tf.shape(pred_boxes)[:2]
            hwhw = tf.tile(tf.cast(hw, tf.float32), [2])
            pred_boxes = tf.reshape(pred_boxes, [hw[0] * hw[1], 4])
            pred_labels = tf.reshape(pred_labels, [hw[0] * hw[1], num_classes])

            pred_probs = tf.nn.sigmoid(pred_labels)

            neg_label_cost = (1 - self._alpha) * tf.pow(pred_probs, self._gamma) * (-tf.math.log(1 - pred_probs + 1e-8))
            pos_label_cost = self._alpha * tf.pow(1 - pred_probs, self._gamma) * (-tf.math.log(pred_probs + 1e-8))
            label_cost = tf.gather(neg_label_cost, gt_labels, axis=-1) - tf.gather(pos_label_cost, gt_labels, axis=-1)

            # Compute the L1 cost between boxes
            bbox_cost = tf.reduce_sum(tf.abs(tf.expand_dims(pred_boxes, 1) / hwhw - tf.expand_dims(gt_boxes, 0) / hwhw), 2)

            # Comput the IoU cost between boxes
            giou_cost = compute_unaligned_iou(gt_boxes, pred_boxes, self._iou_type)

            cost = self._class_weights * label_cost + self._l1_weight * bbox_cost + self._iou_weight * giou_cost

            inds = tf.argmin(cost, 0)

            tgt_boxes = tf.zeros_like(pred_boxes)
            tgt_labels = tf.zeros_like(pred_labels)

            tgt_boxes = tf.tensor_scatter_nd_update(tgt_boxes, inds[:, None], gt_boxes)
            tgt_labels = tf.tensor_scatter_nd_update(tgt_labels, inds[:, None], tf.one_hot(gt_labels, num_classes))
            
            return tgt_boxes, tgt_labels
            
    def __call__(self, gt_boxes, gt_labels, pred_boxes, pred_labels):
        return self.assign(gt_boxes, gt_labels, pred_boxes, pred_labels)


def test():
    import numpy as np

    pred_boxes = tf.random.uniform([64, 64, 4], 0, 255)
    pred_labels = tf.random.uniform([64, 64, 80], -5., 5.)

    gt_boxes = tf.constant([[32, 120, 120, 256], [200, 201, 434, 472]], tf.float32)
    gt_labels = tf.constant([1, 23], tf.int32)
    
    assigner = MinCostAssigner()
    boxes, labels = assigner(gt_boxes, gt_labels, pred_boxes, pred_labels)
    print(gt_boxes)
    print(tf.gather_nd(boxes, tf.where(boxes > 0)))
    print(tf.where(labels == 1))
    

if __name__ == "__main__":
    test()

