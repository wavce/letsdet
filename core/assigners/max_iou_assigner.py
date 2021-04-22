import tensorflow as tf
from utils import box_utils
from .assigner import Assigner
from ..builder import ASSIGNERS 


@ASSIGNERS.register
class MaxIoUAssigner(Assigner):
    """Assign a corresponding gt bbox or background to each bbox.

    Each proposals will be assigned with `-1`, `0`, or a positive integer
    indicating the ground truth index:
        - -1: don't care
        - 0: negative sample, no assigned gt
        - positive integer: positive sample, index (1-based) of assigned gt
    Args:
        pos_iou_thresh (float): IoU threshold for positive boxes.
        neg_iou_thresh (float or tuple): IoU threshold for negative boxes.
    """
    def __init__(self,
                 pos_iou_thresh,
                 neg_iou_thresh=None,
                 min_pos_iou=0.,
                 assign_all_max_gt_overlaps=True,
                 match_low_quality=True,
                 dtype=tf.float32):
        super(MaxIoUAssigner, self).__init__(dtype)

        pos_iou_thresh = pos_iou_thresh
        neg_iou_thresh = neg_iou_thresh if neg_iou_thresh is not None else pos_iou_thresh
        if isinstance(neg_iou_thresh, (int, float)):
            neg_iou_thresh = [0., neg_iou_thresh]

        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.assign_all_max_gt_overlaps = assign_all_max_gt_overlaps
        self.min_pos_iou = min_pos_iou
        self.match_low_quality = match_low_quality

    def assign(self, gt_boxes, gt_labels, proposals):
        """Assign gt to boxes/

        This method assign a gt box to every box (proposal/anchor), each box
        will be assigned with -1, 0 or a positive number. -1 means don't care,
        0 means negative sample, positive number is the index (1-based) of
        assigned gt.

        The assignment is done in following steps, the order matters:
        1. initialize target boxes and labels.
        2. assign proposals whose iou with all gts < neg_iou_thresh to  0.
        3. for each box, if the iou with its nearest gt >= pos_iou_thresh,
            assign it to that box.
        4. for each gt box, assign its best proposals (may be more than
            one) to itself.

        Args:
            proposals (Tensor): Bounding boxes to be assigned, shape (n, 4).
            gt_boxes (Tensor): Ground-truth boxes, shape (k, 4).
            gt_labels (Tensor): Ground-truth labels, shape (k, ).

        Returns:
            target_boxes (Tensor), target_labels (Tensor).
        """
        valid_mask = tf.greater(gt_labels, 0)
        gt_boxes = tf.boolean_mask(gt_boxes, valid_mask)
        gt_labels = tf.boolean_mask(gt_labels, valid_mask)
        gt_boxes = tf.concat([gt_boxes, tf.zeros([1, 4], gt_boxes.dtype)], axis=0)
        gt_labels = tf.concat([gt_labels, tf.zeros([1], gt_labels.dtype)], axis=0)
        overlaps = box_utils.bbox_overlap(proposals, gt_boxes)

        # if tf.greater(self.ignore_iof_thresh, 0) and ignored_gt_boxes is not None:
        #     ignored_overlaps = unaligned_box_iof(proposals, ignored_gt_boxes)   # (n, m)
        #     ignored_max_overlaps = tf.reduce_max(ignored_overlaps, axis=1, keepdims=True)  # (n, )
        
        #     ignored_mask = tf.greater_equal(ignored_max_overlaps, self.ignore_iof_thresh)
        #     overlaps *= tf.cast(ignored_mask, overlaps.dtype)

        return self.assign_wrt_overlaps(overlaps, gt_boxes, gt_labels)

    def assign_wrt_overlaps(self, overlaps, gt_boxes, gt_labels):
        """Assign w.r.t. the overlaps of boxes with gts.

        Args:
            overlaps (Tensor): Overlaps between k gt_boxes and n proposals,
                shape (n, k).
            gt_boxes (Tensor): Ground-truth boxes, shape (k, 4).
            gt_labels (Tensor): Ground-truth labels, shape (k, ).

        Returns:
            target_boxes (Tensor), target_labels (Tensor).
        """
        # num_gts = tf.shape(overlaps)[1]
        num_proposals = tf.shape(overlaps)[0]
        target_labels = -1 * tf.ones([num_proposals], gt_labels.dtype)

        # 1. assign init target boxes and labels
        max_overlaps = tf.reduce_max(overlaps, axis=1)
        argmax_overlaps = tf.argmax(overlaps, axis=1, name="argmax_overlaps")
        target_boxes = tf.gather(gt_boxes, indices=argmax_overlaps)
        
        # 2. assign negative
        negative = tf.logical_and(x=tf.greater_equal(max_overlaps, self.neg_iou_thresh[0]),
                                  y=tf.less(max_overlaps, self.neg_iou_thresh[1]))
        target_labels = tf.where(negative, tf.zeros_like(target_labels), target_labels)

        positive = tf.greater_equal(max_overlaps, self.pos_iou_thresh)
        target_labels = tf.where(positive, tf.gather(gt_labels, indices=argmax_overlaps), target_labels)
                
        # 3. assign the best anchor to gt
        if self.match_low_quality:
            for i in tf.range(tf.shape(gt_boxes)[0]):
                best_overlap = tf.reduce_max(overlaps[:, i])
                if best_overlap > self.min_pos_iou:
                    if self.assign_all_max_gt_overlaps:
                        best_indices = tf.where(overlaps[:, i] == best_overlap)
                        best_gt_boxes = tf.tile(gt_boxes[i:i+1], [tf.shape(best_indices)[0], 1])
                        best_gt_labels = tf.tile(gt_labels[i:i+1], [tf.shape(best_indices)[0]])
                      
                        target_boxes = tf.tensor_scatter_nd_update(target_boxes, best_indices, best_gt_boxes)
                        target_labels = tf.tensor_scatter_nd_update(target_labels, best_indices, best_gt_labels)
                    else:
                        best_indices = tf.argmax(overlaps[:, i])
                        target_boxes = tf.tensor_scatter_nd_update(target_boxes, [[best_indices]], gt_boxes[i:i+1])
                        target_labels = tf.tensor_scatter_nd_update(target_labels, [[best_indices]], gt_labels[i:i+1])

        target_boxes = tf.stop_gradient(target_boxes)
        target_labels = tf.stop_gradient(target_labels)

        return target_boxes, target_labels
   