import tensorflow as tf


class Assigner(object):
    def __init__(self, dtype=tf.float32):

        self.dtype = dtype

    @property
    def _param_dtype(self):
        if self.dtype == tf.float16 or self.dtype == tf.bfloat16:
            return tf.float32

        return self.dtype or tf.float32

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
        raise NotImplementedError()

    def assign_wrt_overlaps(self, overlaps, gt_boxes, gt_labels):
        """Assign w.r.t. the overlaps of boxes with gts.

        Args:
            overlaps (Tensor): Overlaps between k gt_boxes and n proposals,
                shape (k, n).
            gt_boxes (Tensor): Ground-truth boxes, shape (k, 4).
            gt_labels (Tensor): Ground-truth labels, shape (k, ).

        Returns:
            target_boxes (Tensor), target_labels (Tensor).
        """
        raise NotImplementedError()
    
    def __call__(self, gt_boxes, gt_labels, proposals):
        with tf.name_scope("max_iou_assigner"):
            return self.assign(gt_boxes, gt_labels, proposals)