import numpy as np
import tensorflow as tf
from ..builder import METRICS


@METRICS.register
class WiderFaceAP(tf.keras.metrics.Metric):
    def __init__(self, iou_threshold=0.5, **kwargs):
        super(WiderFaceAP, self).__init__(**kwargs)
        self.iou_threshold = iou_threshold
        self.num_thresh = 1000

        self.easy_gt_list = []
        self.medium_gt_list = []
        self.hard_gt_list = []
        self.pred_boxes_list = []
        self.pred_scores_list = []
        self.gt_boxes_list = []
    
    def norm_score(self, pred_scores_list):
        max_score = 0
        min_score = 1
        for score in pred_scores_list:
            if len(score) <= 0:
                continue
            max_score = max(np.max(score), max_score)
            min_score = min(np.min(score), min_score)
        
        results = []
        diff = max_score - min_score
        for score in pred_scores_list:
            if len(score) <= 0:
                results.append(score)
                continue
            
            score = (score - min_score) / diff
            results.append(score)
        
        return results
    
    def unaligned_box_iou(self, boxes1, boxes2):
        """Calculate overlap between two set of unaligned boxes.
            'unaligned' mean len(boxes1) != len(boxes2).

            Args:
                boxes1 (tensor): shape (n, 4).
                boxes2 (tensor): shape (m, 4), m not equal n.

            Returns:
                ious (Tensor): shape (m, n)
        """
        boxes1 = boxes1[:, None, :]   # (n, 1, 4)
        boxes2 = boxes2[None, :, :]   # (1, m, 4)
        
        area1 = (boxes1[..., 2] - boxes1[..., 0] + 1) * (boxes1[..., 3] - boxes1[..., 1] + 1)  # (n, 1)
        area2 = (boxes2[..., 2] - boxes2[..., 0] + 1) * (boxes2[..., 3] - boxes2[..., 1] + 1)  # (1, m)
        inter_y1 = np.maximum(boxes1[..., 0], boxes2[..., 0])  # (n, m)
        inter_x1 = np.maximum(boxes1[..., 1], boxes2[..., 1])  # (n, m)
        inter_y2 = np.minimum(boxes1[..., 2], boxes2[..., 2])  # (n, m)
        inter_x2 = np.minimum(boxes1[..., 3], boxes2[..., 3])  # (n, m)
        inter_h = np.maximum(0., inter_y2 - inter_y1 + 1.)
        inter_w = np.maximum(0., inter_x2 - inter_x1 + 1.)
        inter_area = inter_h * inter_w
        ious = inter_area / (area1 + area2 - inter_area)

        return ious
        
    def image_eval(self, pred_boxes, gt_boxes, ignore):
        pred_recall = np.zeros(pred_boxes.shape[0])
        recall_list = np.zeros(gt_boxes.shape[0])
        proposal_list = np.ones(pred_boxes.shape[0])

        ious = self.unaligned_box_iou(pred_boxes, gt_boxes)
        
        for h in range(pred_boxes.shape[0]):
            gt_ious = ious[h]
            max_iou, max_idx = gt_ious.max(), gt_ious.argmax()
            
            if max_iou >= self.iou_threshold:
                if ignore[max_idx] == 0:
                    recall_list[max_idx] = -1
                    proposal_list[h] = -1
                elif recall_list[max_idx] == 0:
                    recall_list[max_idx] = 1
            
            r_keep_index = np.where(recall_list == 1)[0]
            pred_recall[h] = len(r_keep_index)
        
        return pred_recall, proposal_list
    
    def img_pr_info(self, pred_boxes, pred_scores, proposal_list, pred_recall):
        pr_info = np.zeros((self.num_thresh, 2)).astype("float")

        for t in range(self.num_thresh):
            thresh = 1 - (t + 1) / self.num_thresh
            r_index = np.where(pred_scores >= thresh)[0]
            if len(r_index) == 0:
                pr_info[t, 0] = 0
                pr_info[t, 1] = 0
            else:
                r_index = r_index[-1]
                p_index = np.where(proposal_list[:r_index + 1] == 1)[0]
                pr_info[t, 0] = len(p_index)
                pr_info[t, 1] = pred_recall[r_index]
        
        return pr_info
    
    def dataset_pr_info(self, pr_curve, count_face):
        _pr_curve = np.zeros((self.num_thresh, 2))
        for i in range(self.num_thresh):
            _pr_curve[i, 0] = pr_curve[i, 1] / pr_curve[i, 0]
            _pr_curve[i, 1] = pr_curve[i, 1] / count_face
        return _pr_curve
    
    def voc_ap(self, rec, prec):
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

        return ap

    def update_state(self, pred_boxes, pred_scores, gt_boxes, easy_gt, medium_gt, hard_gt):
        batch_size = tf.shape(gt_boxes)[0]
        for b in tf.range(batch_size):
            valid_gt_mask = tf.logical_not(tf.reduce_all(gt_boxes[b] <= 0, 1))
            valid_pred_mask = tf.logical_not(tf.reduce_all(pred_boxes[b] <= 0, 1))

            valid_pred_boxes = tf.boolean_mask(pred_boxes[b], valid_pred_mask)
            valid_pred_scores = tf.boolean_mask(pred_scores[b], valid_pred_mask)
            valid_gt_boxes = tf.boolean_mask(gt_boxes[b], valid_gt_mask)
            valid_easy_gt = tf.boolean_mask(easy_gt[b], valid_gt_mask)
            valid_medium_gt = tf.boolean_mask(medium_gt[b], valid_gt_mask)
            valid_hard_gt = tf.boolean_mask(hard_gt[b], valid_gt_mask)

            self.pred_boxes_list.append(valid_pred_boxes.numpy())
            self.pred_scores_list.append(valid_pred_scores.numpy())
            self.gt_boxes_list.append(valid_gt_boxes.numpy())
            self.easy_gt_list.append(valid_easy_gt.numpy())
            self.medium_gt_list.append(valid_medium_gt.numpy())
            self.hard_gt_list.append(valid_hard_gt.numpy())

    def result(self):
        pred_scores_list = self.norm_score(self.pred_scores_list)
        pred_boxes_list = self.pred_boxes_list
        gt_boxes_list = self.gt_boxes_list
        # settings = ["easy", "medium", "hard"]
        settings_gt = [self.easy_gt_list, self.medium_gt_list, self.hard_gt_list]
        aps = []

        for gt_list in settings_gt:
            count_face = 0
            pr_curve = np.zeros((self.num_thresh, 2)).astype("float")
            for i in range(len(pred_scores_list)):
                ignore = gt_list[i]
                count_face += np.sum(ignore)
                pred_boxes = pred_boxes_list[i]
                pred_scores = pred_scores_list[i]
                gt_boxes = gt_boxes_list[i]

                if len(gt_boxes) == 0 or len(pred_boxes) == 0:
                    continue

                pred_recall, proposal_list = self.image_eval(pred_boxes, gt_boxes, ignore)
                _img_pr_info = self.img_pr_info(pred_boxes, pred_scores, proposal_list, pred_recall)
                pr_curve += _img_pr_info
            
            pr_curve = self.dataset_pr_info(pr_curve, count_face)
            precision = pr_curve[:, 0]
            recall = pr_curve[:, 1]

            ap = self.voc_ap(recall, precision)
            aps.append(ap)

        print("==================== Results ====================")
        print("Easy   Val AP: {}".format(aps[0]))
        print("Medium Val AP: {}".format(aps[1]))
        print("Hard   Val AP: {}".format(aps[2]))
        print("=================================================")

        return tf.convert_to_tensor(aps, tf.float32)
    
    def reset_states(self):
        self.easy_gt_list = []
        self.medium_gt_list = []
        self.hard_gt_list = []
        self.pred_boxes_list = []
        self.pred_scores_list = []
        self.gt_boxes_list = []

        super(WiderFaceAP, self).reset_states()

