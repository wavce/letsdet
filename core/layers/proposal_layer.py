import tensorflow as tf 


def _single_level_rois_select(boxes, scores, topk, max_nms_outputs, nms_threshold):
    boxes = tf.cast(boxes, tf.float32)
    scores = tf.cast(scores, tf.float32)
    scores = tf.squeeze(scores, -1)

    topk_scores, topk_indices = tf.nn.top_k(scores, k=topk)
    topk_indices = tf.stack(
        [tf.tile(tf.range(tf.shape(boxes)[0])[:, None], [1, tf.shape(topk_scores)[1]]), topk_indices], -1)
    topk_boxes = tf.gather_nd(boxes, topk_indices)

    nmsed_boxes, nmsed_scores, _, _ = tf.image.combined_non_max_suppression(
        tf.expand_dims(topk_boxes, -2), 
        tf.expand_dims(topk_scores, -1), 
        max_nms_outputs, 
        max_nms_outputs, 
        nms_threshold)

    return nmsed_boxes, nmsed_scores


class ProposalLayer(tf.keras.layers.Layer):
    def __init__(self, pre_nms_size=12000, post_nms_size=2000, max_total_size=2000, iou_threshold=0.7, min_size=0, **kwargs):
        super(ProposalLayer, self).__init__(**kwargs)

        self.min_size = min_size
        self.nms_pre = pre_nms_size
        self.nms_post = post_nms_size
        self.iou_threshold = iou_threshold
        self.max_total_size = max_total_size
    
    def call(self, boxes, scores):
        selected_boxes, selected_scores = _single_level_rois_select(
            boxes, scores, self.nms_pre, self.max_total_size, self.iou_threshold)

        return selected_boxes[:, :self.nms_post], selected_scores[:, :self.nms_post]
    
    def compute_output_shape(self, input_shape):
        return tf.TensorShape([input_shape[0], self.nms_post, 4])
