import numpy as np 
import tensorflow  as tf
from tensorflow.python.keras.backend import dtype 
from ..builder import METRICS


@METRICS.register
class mAP(tf.keras.metrics.Metric):
    def __init__(self, num_classes, metric="interp", **kwargs):
        super(mAP, self).__init__(**kwargs)

        assert tf.executing_eagerly()

        self.num_classes = num_classes
        self.iou_threshold = tf.linspace(0.5, 0.95, 10)
        self.predicted_scores = []
        self.predicted_classes = []
        self.gt_classes = []
        self.total_correct = []
        self.metric = "interp"
    
    def _compute_ap(self, recall, precision):
        """ Compute the average precision, given the recall and precision curves.
            Source: https://github.com/rbgirshick/py-faster-rcnn.
            
            Args
                recall:    The recall curve .
                precision: The precision curve .
            
            Returns
                The average precision as computed in py-faster-rcnn.
        """
        # Append sentinel values to beginning and end
        mrec = np.concatenate(([0.], recall, [min(recall[-1] + 1e-3, 1.)]))
        mpre = np.concatenate(([0.], precision, [0.]))

        # Compute the precision envelope
        mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
        
        if self.metric == "interp":
            # 101 point metric (COCO)
            x = np.linspace(0.0, 1, 101)
            ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate

        else: # "continuous"
            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            inds = np.where(mrec[1:] != mrec[:-1])[0]
            # and sum (\Delta recall) * prec
            ap = np.sum((mrec[inds + 1] - mrec[inds]) * mpre[inds + 1])

        return ap
    
    def unaligned_box_iou(self, boxes1, boxes2):
        """Calculate overlap between two set of unaligned boxes.
            'unaligned' mean len(boxes1) != len(boxes2).

            Args:
                boxes1 (tensor): shape (n, 4).
                boxes2 (tensor): shape (m, 4), m not equal n.

            Returns:
                ious (Tensor): shape (n, m)
        """
        boxes1 = boxes1[:, None, :]   # (n, 1, 4)
        boxes2 = boxes2[None, :, :]   # (1, m, 4)
        lt = tf.math.maximum(boxes1[..., 0:2], boxes2[..., 0:2])  # (n, m, 2)
        rb = tf.math.minimum(boxes1[..., 2:4], boxes2[..., 2:4])  # (n, m, 2)

        wh = tf.math.maximum(0.0, rb - lt)  # (n, m, 2)
        overlap = wh[..., 0] * wh[..., 1]  # (n, m)
        area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1] )  # (n, m)
        area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1] )  # (n, m)

        ious = overlap / (area1 + area2 - overlap)

        return ious
    
    def _get_correct(self, pred_boxes, pred_classes, gt_boxes, gt_classes, gt_boxes_ignore=None):
        """Check if predicted boxes are true positive or false positive.
        
        Args:
            pred_boxes (tensor): Predicted boxes of this image, of shape (m, 4).
            pred_classes (tensor): Predicted classes of boxes in the image, of shape (m, )
            gt_boxes (tensor): GT boxes of this image, of shape (n, 4).
            gt_classes (tensor): GT classes of boxes in the image, of shape (m, )
            gt_boxes_ignore (tensor): Ignored gt boxes of this image, of shape (k, 4).
                Default to None.
        
        Returns:
            tensor: the per box in predictions whether is true.
        """
        # an indicator of ignored gts
        # gt_ignore_inds = tf.zeros(gt_boxes.shape[0], dtype=tf.bool)
        # if gt_boxes_ignore is not None:
        #     gt_ignore_inds = tf.concat(
        #         [gt_ignore_inds, tf.ones(gt_boxes_ignore.shape[0], dtype=tf.bool)], 0)

        #     # stack gt_boxes and gt_boxes_ignore for convience
        #     gt_boxes = np.stack([gt_boxes, gt_boxes_ignore], 0)
        
        num_preds = tf.shape(pred_boxes)[0]
        num_gts = tf.shape(gt_boxes)[0]
        correct = tf.zeros((num_preds, tf.size(self.iou_threshold)), dtype=tf.float32)
        detected = tf.zeros([num_gts], dtype=tf.int32)
        unique_cls, _ = tf.unique(gt_classes)
        for c in unique_cls:
            ti = tf.where(tf.cast(gt_classes, c.dtype) == c)
            pi = tf.where(tf.cast(pred_classes, c.dtype) == c) 
            if tf.size(pi) > 0:
                tboxes = tf.gather_nd(gt_boxes, ti)
                pboxes = tf.gather_nd(pred_boxes, pi)
                ious = self.unaligned_box_iou(pboxes, tboxes)
                max_ious = tf.reduce_max(ious, 1)
                max_iou_inds = tf.argmax(ious, 1)
                
                for i in tf.squeeze(tf.where(max_ious > self.iou_threshold[0]), -1):
                    d = ti[max_iou_inds[i]]
                    if detected[d[0]] == 0:
                        detected = tf.tensor_scatter_nd_update(detected, [d], [1])
                        v = tf.cast(max_ious[i] > self.iou_threshold, correct.dtype)
                        correct = tf.tensor_scatter_nd_update(correct, [pi[i]], [v])

        return correct

    def update_state(self, 
                     gt_boxes, 
                     gt_classes, 
                     pred_boxes, 
                     pred_scores, 
                     pred_classes, 
                     gt_boxes_ignore=None, 
                     sample_weights=None):
        batch_size = tf.shape(gt_boxes)[0]
        for b in tf.range(batch_size):
            valid_gt_mask = gt_classes[b] > 0
            valid_pred_mask = pred_classes[b] > 0

            valid_gt_boxes = tf.boolean_mask(gt_boxes[b], valid_gt_mask)
            valid_gt_classes = tf.boolean_mask(gt_classes[b], valid_gt_mask)
            valid_pred_boxes = tf.boolean_mask(pred_boxes[b], valid_pred_mask)
            valid_pred_classes = tf.boolean_mask(pred_classes[b], valid_pred_mask)
            valid_pred_scores = tf.boolean_mask(pred_scores[b], valid_pred_mask)

            self.gt_classes.append(valid_gt_classes)
            if tf.size(valid_gt_classes) > 0:
                self.total_correct.append(
                    self._get_correct(
                        valid_pred_boxes, valid_pred_classes, valid_gt_boxes, valid_gt_classes, gt_boxes_ignore))
                self.predicted_classes.append(valid_pred_classes)
                self.predicted_scores.append(valid_pred_scores)

    def result(self):
        correct = tf.concat(self.total_correct, 0)
        pred_scores = tf.concat(self.predicted_scores, 0)
        pred_classes = tf.concat(self.predicted_classes, 0)
        gt_classes = tf.concat(self.gt_classes, 0)

        correct = correct.numpy()
        pred_scores = pred_scores.numpy()
        pred_classes = pred_classes.numpy()
        gt_classes = gt_classes.numpy()

        i = np.argsort(-pred_scores)
        correct = correct[i]
        pred_scores = pred_scores[i]
        pred_classes = pred_classes[i]
        
        # Create Precision-Recall curve and compute AP for each class
        pr_score = 0.1  # score to evaluate P and R https://github.com/ultralytics/yolov3/issues/898
        s = [self.num_classes, correct.shape[1]]  # number class, number iou thresholds (i.e. 10 for mAP0.5...0.95)
        ap, p, r = np.zeros(s), np.zeros(s), np.zeros(s)    
        for c in range(self.num_classes):
            i = pred_classes == c
            num_preds =  i.sum()
            num_gts = (gt_classes == c).sum()

            if num_gts == 0 or num_preds == 0:
                continue
            
            # Accumulate FPs and TPs
            fpc = (1. - correct[i]).cumsum(0)
            tpc = correct[i].cumsum(0)

            # Recall 
            recall = tpc / (num_gts + 1e-6)  # recall curve
            r[c] = np.interp(-pr_score, -pred_scores[i], recall[:, 0])  # r at pr_score, negative x, xp because xp decreases

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p[c] = np.interp(-pr_score, -pred_scores[i], precision[:, 0])  

            # AP from recall-precision curve
            for j in range(correct.shape[1]):
                ap[c, j] = self._compute_ap(recall[:, j], precision[:, j])
        
        return tf.convert_to_tensor(ap, tf.float32)
    
    def reset_states(self):
        self.total_correct.clear()
        self.predicted_classes.clear()
        self.predicted_scores.clear()
        self.gt_classes.clear()
        
        super(mAP, self).reset_states()
        

