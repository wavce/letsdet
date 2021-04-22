import tensorflow as tf
from core.builder import NMS


def non_max_suppression(predicted_boxes, 
                        predicted_scores, 
                        num_classes=80, 
                        post_nms_size=100,
                        iou_threshold=0.6,
                        score_threshold=None):
    with tf.name_scope("nms"):
        tmp_boxes_list = []
        tmp_scores_list = []
        tmp_classses_list = []
        predicted_classes = tf.argmax(predicted_scores, -1)
        predicted_scores = tf.reduce_max(predicted_scores, -1)
        for c in range(num_classes):
            current_mask = predicted_classes == c
            current_boxes = tf.boolean_mask(predicted_boxes, current_mask)
            current_scores = tf.boolean_mask(predicted_scores, current_mask)
            current_classes = tf.boolean_mask(predicted_classes, current_mask)
            selected_indices = tf.image.non_max_suppression(
                boxes=current_boxes,
                scores=current_scores,
                max_output_size=post_nms_size,
                iou_threshold=iou_threshold,
                score_threshold=score_threshold)
            selected_boxes = tf.gather(current_boxes, selected_indices)
            selected_scores = tf.gather(current_scores, selected_indices)
            selected_classes = tf.gather(current_classes, selected_indices)

            tmp_boxes_list.append(selected_boxes)
            tmp_scores_list.append(selected_scores)
            tmp_classses_list.append(selected_classes)
            
        boxes = tf.concat(tmp_boxes_list, 0)
        scores = tf.concat(tmp_scores_list, 0)
        classes = tf.concat(tmp_classses_list, 0)
        classes = tf.cast(classes, scores.dtype)

        return boxes, scores, classes


def soft_nms(predicted_boxes, 
             predicted_scores,
             post_nms_size=100,
             iou_threshold=0.6,
             score_threshold=None, 
             soft_nms_sigma=0.5,
             num_classes=80):
    with tf.name_scope("soft_nms"):
        tmp_boxes_list = []
        tmp_scores_list = []
        tmp_classses_list = []
        predicted_classes = tf.argmax(predicted_scores, -1)
        predicted_scores = tf.reduce_max(predicted_scores, -1)
        for c in range(num_classes):
            current_mask = predicted_classes == c
            current_boxes = tf.boolean_mask(predicted_boxes, current_mask)
            current_scores = tf.boolean_mask(predicted_scores, current_mask)
            current_classes = tf.boolean_mask(predicted_classes, current_mask)
            selected_indices, _ = tf.image.non_max_suppression_with_scores(
                boxes=current_boxes,
                scores=current_scores,
                max_output_size=post_nms_size,
                iou_threshold=iou_threshold,
                score_threshold=score_threshold,
                soft_nms_sigma=soft_nms_sigma)
            selected_boxes = tf.gather(current_boxes, selected_indices)
            selected_scores = tf.gather(current_scores, selected_indices)
            selected_classes = tf.gather(current_classes, selected_indices)

            tmp_boxes_list.append(selected_boxes)
            tmp_scores_list.append(selected_scores)
            tmp_classses_list.append(selected_classes)
        
        boxes = tf.concat(tmp_boxes_list, 0)
        scores = tf.concat(tmp_scores_list, 0)
        classes = tf.concat(tmp_classses_list, 0)
        classes = tf.cast(classes, scores.dtype)

        return boxes, scores, classes
    

def _unaligned_box_iou_for_fast_nms(self, boxes):
    """Calculate overlap between two set of unaligned boxes.
        'unaligned' mean len(boxes1) != len(boxes2).

        Args:
            boxes (tensor): shape (c, k, 4).
        Returns:
            ious (Tensor): shape (c, k, k)
    """
    boxes1 = boxes[..., :, None, :]   # (c, k, 4)
    boxes2 = boxes[..., None, :, :]   # (c, k, 4)
    inter_y1 = tf.maximum(boxes1[..., 0], boxes2[..., 0])  # (k, k)
    inter_x1 = tf.maximum(boxes1[..., 1], boxes2[..., 1])  # (k, k)
    inter_y2 = tf.minimum(boxes1[..., 2], boxes2[..., 2])  # (k, k)
    inter_x2 = tf.minimum(boxes1[..., 3], boxes2[..., 3])  # (k, k)

    inter_h = tf.maximum(0.0, inter_y2 - inter_y1)  # (k, k)
    inter_w = tf.maximum(0.0, inter_x2 - inter_x1)  # (k, k)
    overlap = inter_h * inter_w
    
    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])  # (k, k)
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])  # (k, k)

    ious = overlap / (area1 + area2 - overlap)

    return ious


def _unaligned_box_iou_for_matrix_nms(boxes):
    """Calculate overlap between two set of unaligned boxes.
        'unaligned' mean len(boxes1) != len(boxes2).

        Args:
            boxes (tensor): shape (k, 4).
        Returns:
            ious (Tensor): shape (k, k)
    """
    boxes1 = boxes[:, None, :]   # (k, 1, 4)
    boxes2 = boxes[None, :, :]   # (1, k, 4)
    inter_y1 = tf.maximum(boxes1[..., 0], boxes2[..., 0])  # (k, k)
    inter_x1 = tf.maximum(boxes1[..., 1], boxes2[..., 1])  # (k, k)
    inter_y2 = tf.minimum(boxes1[..., 2], boxes2[..., 2])  # (k, k)
    inter_x2 = tf.minimum(boxes1[..., 3], boxes2[..., 3])  # (k, k)

    inter_h = tf.maximum(0.0, inter_y2 - inter_y1)  # (k, k)
    inter_w = tf.maximum(0.0, inter_x2 - inter_x1)  # (k, k)
    overlap = inter_h * inter_w
    
    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])  # (k, k)
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])  # (k, k)

    ious = overlap / (area1 + area2 - overlap)

    return ious


def fast_nms(predicted_boxes, 
             predicted_scores, 
             pre_nms_size=200,
             iou_threshold=0.6,
             score_threshold=None):
    with tf.name_scope("fast_nms"):                              # (num_dets, )
        thresh_mask = tf.reduce_max(predicted_scores, -1) > score_threshold                          # (num_dets, )

        thresh_scores = tf.boolean_mask(predicted_scores, thresh_mask)   # (n1, num_classes)
        thresh_boxes = tf.boolean_mask(predicted_boxes, thresh_mask)     # (n1, 4)
        
        sorted_inds = tf.argsort(thresh_scores, 0, direction="DESCENDING")  # (n1, num_classes)
        sorted_scores = tf.sort(thresh_scores, 0, direction="DESCENDING")
        topk_inds = sorted_inds[:pre_nms_size]                              # (topk, num_classes)

        topk_boxes = tf.gather(thresh_boxes, topk_inds)                     # (topk, num_classes, 4)
        topk_scores = sorted_scores[:pre_nms_size]                          # (topk, num_classes)
        topk_boxes = tf.transpose(topk_boxes, [1, 0, 2])
        topk_scores = tf.transpose(topk_scores, [1, 0])

        iou = _unaligned_box_iou_for_fast_nms(topk_boxes)                            # (num_classes, topk, topk)
        iou -= tf.linalg.band_part(iou, -1, 0)
        max_iou = tf.reduce_max(iou, 1)                                     # (num_classes, topk)

        # Now just filter out the ones higher than the threshold
        keep = max_iou <= iou_threshold                            # (num_classes, topk)
        keep = tf.logical_and(keep, topk_scores > score_threshold)
        num_classes = tf.shape(max_iou)[0]
        num_samples = tf.shape(max_iou)[1]
        # Assign each kept detection to its corresponding class
        classes = tf.tile(tf.expand_dims(tf.range(num_classes), 1), [1, num_samples])
        classes = tf.boolean_mask(classes, keep)
        boxes = tf.boolean_mask(topk_boxes, keep)
        scores = tf.boolean_mask(topk_scores, keep)
        classes = tf.cast(classes, scores.dtype)

        return boxes, scores, classes


def matrix_nms(predicted_boxes, 
               predicted_scores, 
               pre_nms_size=200,
               update_threshold=0.6,
               score_threshold=0.3,
               kernel="linear",
               sigma=2.0):
    with tf.name_scope("matrix_nms"):
        inds = tf.where(predicted_scores > score_threshold)             # (num_dets, )
        thresh_scores = tf.gather_nd(predicted_scores, inds)             # (n1, )
        thresh_classes = inds[:, 1]
        thresh_boxes = tf.gather(predicted_boxes, inds[:, 0])      # (n1, 4)
        # sort and keep top pre_nms_size
        sorted_inds = tf.argsort(thresh_scores, direction="DESCENDING")  # (n1, )
        topk_inds = sorted_inds[:pre_nms_size]                         # (n1, )
        topk_boxes = tf.gather(thresh_boxes, topk_inds)                     # (topk, 4)
        topk_scores = tf.gather(thresh_scores, topk_inds)                   # (topk, )
        topk_classes = tf.gather(thresh_classes, topk_inds)                 # (topk, )
        
        iou_matrix = _unaligned_box_iou_for_matrix_nms(topk_boxes)                   # (topk, topk)
        num_samples = tf.shape(iou_matrix)[0]
        iou_matrix -= tf.linalg.band_part(iou_matrix, -1, 0) 
        # class specific matrix
        class_specific_x = tf.tile(topk_classes[None, :], [num_samples, 1])
        class_matrix = tf.cast(class_specific_x == tf.transpose(class_specific_x, [1, 0]), tf.float32)
        class_matrix -= tf.linalg.band_part(class_matrix, -1, 0)
        
        # IoU compensation
        decay_iou = iou_matrix * class_matrix
        compensation_iou = tf.reduce_max(decay_iou, 0)  
        compensation_iou = tf.tile(compensation_iou[:, None], [1, num_samples])
        # matrix nms
        if kernel == "gaussian":
            decay_matrix = tf.math.exp(-1 * sigma * (decay_iou ** 2))
            compensation_matrix = tf.math.exp(-1 * sigma * (compensation_iou ** 2))
            decay_coefficient = tf.reduce_min(decay_matrix / compensation_matrix, 0)
        else:
            decay_matrix = (1 - decay_iou) / (1 - compensation_iou)
            decay_coefficient = tf.reduce_min(decay_matrix, 0)
        
        decay_scores = topk_scores * decay_coefficient
        
        keep = decay_scores > update_threshold
        topk_scores = tf.boolean_mask(topk_scores, keep)
        topk_classes = tf.boolean_mask(topk_classes, keep)
        topk_boxes = tf.boolean_mask(topk_boxes, keep)  
        topk_classes = tf.cast(topk_classes, topk_scores.dtype)  

        return topk_boxes, topk_scores, topk_classes


@NMS.register
class NonMaxSuppression(object):
    def __init__(self, iou_threshold, score_threshold, pre_nms_size, post_nms_size, **kwargs):
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.pre_nms_size = pre_nms_size
        self.post_nms_size = post_nms_size
    
    def __call__(self, predicted_boxes, predicted_scores):
        with tf.name_scope("non_max_suppression"):
            nmsed_boxes_ta = tf.TensorArray(size=0, dynamic_size=True, dtype=predicted_boxes.dtype)
            nmsed_scores_ta = tf.TensorArray(size=0, dynamic_size=True, dtype=predicted_scores.dtype)
            nmsed_classes_ta = tf.TensorArray(size=0, dynamic_size=True, dtype=predicted_scores.dtype)
            num_detections_ta = tf.TensorArray(size=0, dynamic_size=True, dtype=tf.int32)
            
            max_predicted_scores = tf.reduce_max(predicted_scores, -1)
            batch_size = tf.shape(predicted_boxes)[0]

            num_classes = tf.keras.backend.int_shape(predicted_scores)[-1]
            post_nms_size = self.post_nms_size
            for i in tf.range(batch_size):
                boxes, scores, classes = non_max_suppression(
                    predicted_boxes=predicted_boxes[i],
                    predicted_scores=max_predicted_scores[i],
                    num_classes=num_classes,
                    post_nms_size=self.post_nms_size,
                    iou_threshold=self.iou_threshold,
                    score_threshold=self.score_threshold)
                num = tf.size(scores)
                if tf.less(num, post_nms_size):
                    n = self.post_nms_size - num
                    boxes = tf.concat([boxes, tf.zeros([n, 4], boxes.dtype)], 0)
                    scores = tf.concat([scores, tf.zeros([n], scores.dtype)], 0)
                    classes = tf.concat([classes, tf.zeros([n], classes.dtype)], 0)
                else:
                    scores, inds = tf.nn.top_k(scores, k=self.post_nms_size)
                    boxes = tf.gather(boxes, inds)
                    classes = tf.gather(classes, inds)
                    num = tf.constant(self.post_nms_size, num.dtype)
                    
                nmsed_boxes_ta = nmsed_boxes_ta.write(i, boxes)
                nmsed_classes_ta = nmsed_scores_ta.write(i, scores)
                nmsed_classes_ta = nmsed_classes_ta.write(i, classes)
                num_detections_ta = num_detections_ta.write(i, num)
                
            nmsed_boxes = nmsed_boxes_ta.stack()
            nmsed_scores = nmsed_scores_ta.stack()
            nmsed_classes = nmsed_classes_ta.stack()
            num_detections = num_detections_ta.stack()
            
            nmsed_boxes_ta.close()
            nmsed_scores_ta.close()
            nmsed_classes_ta.close()
            num_detections_ta.close()

            return dict(nmsed_boxes=nmsed_boxes,
                        nmsed_scores=nmsed_scores, 
                        nmsed_classes=nmsed_classes, 
                        valid_detections=num_detections)
                

@NMS.register
class FastNonMaxSuppression(object):
    def __init__(self, iou_threshold, score_threshold, pre_nms_size=200, post_nms_size=100, **kwargs):
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.pre_nms_size = pre_nms_size
        self.post_nms_size = post_nms_size

    def __call__(self, predicted_boxes, predicted_scores):
        with tf.name_scope("fast_non_max_suppression"):
            nmsed_boxes_ta = tf.TensorArray(size=0, dynamic_size=True, dtype=predicted_boxes.dtype)
            nmsed_scores_ta = tf.TensorArray(size=0, dynamic_size=True, dtype=predicted_scores.dtype)
            nmsed_classes_ta = tf.TensorArray(size=0, dynamic_size=True, dtype=predicted_scores.dtype)
            num_detections_ta = tf.TensorArray(size=0, dynamic_size=True, dtype=tf.int32)

            batch_size = tf.shape(predicted_boxes)[0]
            for i in tf.range(batch_size):
                boxes, scores, classes = fast_nms(
                    predicted_boxes=predicted_boxes[i], 
                    predicted_scores=predicted_scores[i], 
                    pre_nms_size=self.pre_nms_size, 
                    iou_threshold=self.iou_threshold, 
                    score_threshold=self.score_threshold)
                num_valid = tf.size(scores)
                    
                if num_valid < self.post_nms_size:
                    n_ = self.post_nms_size - num_valid
                    boxes = tf.concat([boxes, tf.zeros([n_, 4], boxes.dtype)], 0)
                    classes = tf.concat([classes, tf.zeros([n_, ], classes.dtype)], 0)
                    scores = tf.concat([scores, tf.zeros([n_, ], scores.dtype)], 0)
                else:
                    boxes = boxes[:self.post_nms_size]
                    classes = classes[:self.post_nms_size]
                    scores = scores[:self.post_nms_size]
                    num_valid = tf.constant(self.post_nms_size, num_valid.dtype)

                nmsed_boxes_ta = nmsed_boxes_ta.write(i, boxes)
                nmsed_scores_ta = nmsed_scores_ta.write(i, scores)
                nmsed_classes_ta = nmsed_classes_ta.write(i, classes)
                num_detections_ta = num_detections_ta.write(i, num_valid)
            
            nmsed_boxes = nmsed_boxes_ta.stack()
            nmsed_scores = nmsed_scores_ta.stack()
            nmsed_classes = nmsed_classes_ta.stack()
            num_detections = num_detections_ta.stack()
            
            nmsed_boxes_ta.close()
            nmsed_scores_ta.close()
            nmsed_classes_ta.close()
            num_detections_ta.close()

            return dict(nmsed_boxes=nmsed_boxes,
                        nmsed_scores=nmsed_scores, 
                        nmsed_classes=nmsed_classes, 
                        valid_detections=num_detections)


@NMS.register
class MatrixNonMaxSuppression(object):
    def __init__(self, score_threshold, update_threshold, kernel="linear", pre_nms_size=500, post_nms_size=100, sigma=2.0, **kwargs):
        self.score_threshold = score_threshold
        self.update_threshold = update_threshold
        self.pre_nms_size = pre_nms_size
        self.post_nms_size = post_nms_size
        self.kernel = kernel
        self.sigma = sigma
    
    def __call__(self, predicted_boxes, predicted_scores):
        with tf.name_scope("matrix_non_max_suppression"):
            nmsed_boxes_ta = tf.TensorArray(size=0, dynamic_size=True, dtype=predicted_boxes.dtype)
            nmsed_scores_ta = tf.TensorArray(size=0, dynamic_size=True, dtype=predicted_scores.dtype)
            nmsed_classes_ta = tf.TensorArray(size=0, dynamic_size=True, dtype=predicted_scores.dtype)
            num_detections_ta = tf.TensorArray(size=0, dynamic_size=True, dtype=tf.int32)

            batch_size = tf.shape(predicted_boxes)[0]
            for i in tf.range(batch_size):           
                # thresholding                          
                boxes, scores, classes = matrix_nms(
                    predicted_boxes=predicted_boxes[i],
                    predicted_scores=predicted_scores[i],
                    pre_nms_size=self.pre_nms_size,
                    update_threshold=self.update_threshold,
                    score_threshold=self.score_threshold,
                    kernel=self.kernel,
                    sigma=self.sigma)
                
                num_ = tf.shape(scores)[0]
                if num_ > self.post_nms_size:
                    scores, nmsed_inds = tf.nn.top_k(scores, k=self.post_nms_size)
                    boxes = tf.gather(boxes, nmsed_inds)
                    classes = tf.gather(classes, nmsed_inds)
                    num_ = tf.constant(self.post_nms_size, num_.dtype)
                else:
                    n = self.post_nms_size - num_
                    scores = tf.concat([scores, tf.zeros([n], scores.dtype)], 0)
                    boxes = tf.concat([boxes, tf.zeros([n, 4], boxes.dtype)], 0)
                    classes = tf.concat([classes, tf.zeros([n], classes.dtype)], 0)
                
                nmsed_boxes_ta = nmsed_boxes_ta.write(i, boxes)
                nmsed_classes_ta = nmsed_classes_ta.write(i, classes)
                nmsed_scores_ta = nmsed_scores_ta.write(i, scores)
                num_detections_ta = num_detections_ta.write(i, num_)
            
            nmsed_boxes = nmsed_boxes_ta.stack()
            nmsed_scores = nmsed_scores_ta.stack()
            nmsed_classes = nmsed_classes_ta.stack()
            num_detections = num_detections_ta.stack()
            
            nmsed_boxes_ta.close()
            nmsed_scores_ta.close()
            nmsed_classes_ta.close()
            num_detections_ta.close()

            return dict(nmsed_boxes=nmsed_boxes,
                        nmsed_scores=nmsed_scores, 
                        nmsed_classes=nmsed_classes, 
                        valid_detections=num_detections)


@NMS.register
class SoftNonMaxSuppression(object):
    def __init__(self, iou_threshold, score_threshold, pre_nms_size, post_nms_size, sigma=0.5, **kwargs):
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.pre_nms_size = pre_nms_size
        self.post_nms_size = post_nms_size
        self.soft_nms_sigma = sigma
    
    def __call__(self, predicted_boxes, predicted_scores):
        with tf.name_scope("soft_non_max_suppression"):
            nmsed_boxes_ta = tf.TensorArray(size=0, dynamic_size=True, dtype=predicted_boxes.dtype)
            nmsed_scores_ta = tf.TensorArray(size=0, dynamic_size=True, dtype=predicted_scores.dtype)
            nmsed_classes_ta = tf.TensorArray(size=0, dynamic_size=True, dtype=predicted_scores.dtype)
            num_detections_ta = tf.TensorArray(size=0, dynamic_size=True, dtype=tf.int32)
            
            num_classes = tf.keras.backend.int_shape(predicted_scores)[-1]
            batch_size = tf.shape(predicted_boxes)[0]
            for i in tf.range(batch_size):
                boxes, scores, classes = soft_nms(
                    predicted_boxes=predicted_boxes[i],
                    predicted_scores=predicted_scores[i],
                    post_nms_size=self.post_nms_size,
                    iou_threshold=self.iou_threshold,
                    score_threshold=self.score_threshold,
                    soft_nms_sigma=self.soft_nms_sigma,
                    num_classes=num_classes)

                num = tf.size(scores)
                if tf.less(num, self.post_nms_size):
                    n = self.post_nms_size - num
                    boxes = tf.concat([boxes, tf.zeros([n, 4], boxes.dtype)], 0)
                    scores = tf.concat([scores, tf.zeros([n], scores.dtype)], 0)
                    classes = tf.concat([classes, tf.zeros([n], classes.dtype)], 0)
                else:
                    scores, inds = tf.nn.top_k(scores, k=self.post_nms_size)
                    boxes = tf.gather(boxes, inds)
                    classes = tf.gather(classes, inds)
                    num = tf.constant(self.post_nms_size, num.dtype)
                
                nmsed_boxes_ta = nmsed_boxes_ta.write(i, boxes)
                nmsed_scores_ta = nmsed_scores_ta.write(i, scores)
                nmsed_classes_ta = nmsed_classes_ta.write(i, classes)
                num_detections_ta = num_detections_ta.write(i, num)
            
            nmsed_boxes = nmsed_boxes_ta.stack()
            nmsed_scores = nmsed_scores_ta.stack()
            nmsed_classes = nmsed_classes_ta.stack()
            num_detections = num_detections_ta.stack()
            
            nmsed_boxes_ta.close()
            nmsed_scores_ta.close()
            nmsed_classes_ta.close()
            num_detections_ta.close()

            return dict(nmsed_boxes=nmsed_boxes,
                        nmsed_scores=nmsed_scores, 
                        nmsed_classes=nmsed_classes, 
                        valid_detections=num_detections)
                

@NMS.register
class CombinedNonMaxSuppression(object):
    def __init__(self, iou_threshold, score_threshold, pre_nms_size, post_nms_size, **kwargs):
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.pre_nms_size = pre_nms_size
        self.post_nms_size = post_nms_size
    
    def __call__(self, predicted_boxes, predicted_scores):
        with tf.name_scope("combined_non_max_suppression"):
            # max_predicted_scores = tf.reduce_max(predicted_scores, -1)
            # thresholded_boxes, thresholded_scores = batch_threshold(
            #     predicted_boxes, predicted_scores, max_predicted_scores,
            #     self.score_threshold, self.pre_nms_size)

            predicted_boxes = tf.cond(tf.equal(tf.rank(predicted_boxes), 4),
                                      lambda: predicted_boxes,
                                      lambda: tf.expand_dims(predicted_boxes, 2)) 
            nmsed_boxes, nmsed_scores, nmsed_classes, num_detections = tf.image.combined_non_max_suppression(
                boxes=predicted_boxes,
                scores=predicted_scores,
                max_output_size_per_class=self.post_nms_size,
                max_total_size=self.post_nms_size,
                iou_threshold=self.iou_threshold,
                score_threshold=self.score_threshold,
                clip_boxes=False)
            
            return dict(nmsed_boxes=nmsed_boxes,
                        nmsed_scores=nmsed_scores, 
                        nmsed_classes=nmsed_classes, 
                        valid_detections=num_detections)


@NMS.register
class NonMaxSuppressionWithQuality(object):
    def __init__(self,
                 pre_nms_size=400,
                 post_nms_size=100, 
                 iou_threshold=0.6, 
                 score_threshold=0.5,
                 sigma=0.5,
                 nms_type="nms"):
        self.post_nms_size = post_nms_size
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.sigma = sigma
        self.nms_type = nms_type
        self.pre_nms_size = pre_nms_size
        
        assert nms_type in ["nms", "soft_nms", "fast_nms", "matrix_nms"]
        
    def nms_with_quality(self,
                         boxes, 
                         scores, 
                         quality, 
                         num_classes=80):
        with tf.name_scope("nms_with_quality"):
            quality = tf.squeeze(quality, -1)
            mask = quality > self.score_threshold
            thr_boxes = tf.boolean_mask(boxes, mask)
            thr_scores = tf.boolean_mask(scores, mask)
            thr_quality = tf.boolean_mask(quality, mask)
            
            if tf.greater(tf.size(thr_quality), self.pre_nms_size):
                thr_quality, inds = tf.nn.top_k(thr_quality, k=self.pre_nms_size)
                thr_boxes = tf.gather(thr_boxes, inds)
                thr_scores = tf.gather(thr_scores, inds)
                
            thr_scores = thr_scores * tf.expand_dims(thr_quality, -1)
            if self.nms_type == "nms":
                return non_max_suppression(predicted_scores=thr_scores, 
                                           predicted_boxes=thr_boxes,
                                           num_classes=num_classes,
                                           post_nms_size=self.post_nms_size,
                                           iou_threshold=self.iou_threshold,
                                           score_threshold=0.)
            elif self.nms_type == "soft_nms":
                return soft_nms(predicted_boxes=thr_boxes,
                                predicted_scores=thr_scores,
                                post_nms_size=self.post_nms_size,
                                iou_threshold=self.iou_threshold,
                                score_threshold=0.,
                                soft_nms_sigma=self.sigma,
                                num_classes=num_classes)
            elif self.nms_type == "fast_nms":
                return fast_nms(predicted_boxes=thr_boxes,
                                predicted_scores=thr_scores,
                                pre_nms_size=self.pre_nms_size,
                                iou_threshold=self.iou_threshold,
                                score_threshold=0.)
            elif self.nms_type == "matrix_nms":
                return matrix_nms()
    
    def _nms(self, predicted_boxes, predicted_scores, predicted_quality):
        nmsed_boxes_ta = tf.TensorArray(size=0, dynamic_size=True, dtype=predicted_boxes.dtype)
        nmsed_scores_ta = tf.TensorArray(size=0, dynamic_size=True, dtype=predicted_scores.dtype)
        nmsed_classes_ta = tf.TensorArray(size=0, dynamic_size=True, dtype=predicted_scores.dtype)
        num_detections_ta = tf.TensorArray(size=0, dynamic_size=True, dtype=tf.int32)
        
        batch_size = tf.shape(predicted_boxes)[0]

        num_classes = tf.keras.backend.int_shape(predicted_scores)[-1]
        post_nms_size = self.post_nms_size
        for i in tf.range(batch_size):
            boxes, scores, classes = self.nms_with_quality(
                boxes=predicted_boxes[i],
                scores=predicted_scores[i],
                quality=predicted_quality[i],
                num_classes=num_classes)
            num = tf.size(scores)
            if tf.less(num, post_nms_size):
                n = self.post_nms_size - num
                boxes = tf.concat([boxes, tf.zeros([n, 4], boxes.dtype)], 0)
                scores = tf.concat([scores, tf.zeros([n], scores.dtype)], 0)
                classes = tf.concat([classes, tf.zeros([n], classes.dtype)], 0)
            else:
                scores, inds = tf.nn.top_k(scores, k=self.post_nms_size)
                boxes = tf.gather(boxes, inds)
                classes = tf.gather(classes, inds)
                num = tf.constant(self.post_nms_size, num.dtype)
                
            nmsed_boxes_ta = nmsed_boxes_ta.write(i, boxes)
            nmsed_scores_ta = nmsed_scores_ta.write(i, scores)
            nmsed_classes_ta = nmsed_classes_ta.write(i, classes)
            num_detections_ta = num_detections_ta.write(i, num)
            
        nmsed_boxes = nmsed_boxes_ta.stack()
        nmsed_scores = nmsed_scores_ta.stack()
        nmsed_classes = nmsed_classes_ta.stack()
        num_detections = num_detections_ta.stack()
        
        nmsed_boxes_ta.close()
        nmsed_scores_ta.close()
        nmsed_classes_ta.close()
        num_detections_ta.close()

        return dict(nmsed_boxes=nmsed_boxes,
                    nmsed_scores=nmsed_scores, 
                    nmsed_classes=nmsed_classes, 
                    valid_detections=num_detections)
    
    def __call__(self, predicted_boxes, predicted_scores, predicted_quality):
        return self._nms(predicted_boxes, predicted_scores, predicted_quality)
