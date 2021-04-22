import math
import tensorflow as tf


def _get_v(b1_height, b1_width, b2_height, b2_width):
    """Get the consistency measurement of aspect ratio for ciou."""

    @tf.custom_gradient
    def _get_grad_v(height, width):
        """backpropogate gradient."""
        arctan = (tf.atan(tf.math.divide_no_nan(b1_width, b1_height)) - 
                  tf.atan(tf.math.divide_no_nan(width, height)))
        v = 4 * ((arctan / math.pi) ** 2)

        def _grad_v(dv):
            """Grad for eager mode."""
            gdw = dv * 8 * arctan * height / (math.pi**2)
            gdh = -dv * 8 * arctan * width / (math.pi**2)
            return [gdh, gdw]

        # def _grad_v_graph(dv, variables):
        #     """Grad for graph mode."""
        #     gdw = dv * 8 * arctan * height / (math.pi ** 2)
        #     gdh = -dv * 8 * arctan * width / (math.pi ** 2)

        #     return [gdh, gdw], tf.gradients(v, variables, grad_ys=dv)
      
        return v, _grad_v

    return _get_grad_v(b2_height, b2_width)


def compute_iou(target_boxes, predicted_boxes, iou_type="iou"):
    """Computing the IoU for aligned boxes.
    
    Args:
        predicted_boxes: predicted boxes, with coordinate [y_min, x_min, y_max, x_max].
        target_boxes: target boxes, with coordinate [y_min, x_min, y_max, x_max].
        iou_type: one of ['iou', 'ciou', 'diou', 'giou'].
    Returns:
        IoU loss float `Tensor`.
    """
    iou_type = iou_type.lower()
    assert iou_type in ["iou", "ciou", "diou", "giou"]
    t_y1, t_x1, t_y2, t_x2 = tf.unstack(target_boxes, num=4, axis=-1)
    p_y1, p_x1, p_y2, p_x2 = tf.unstack(predicted_boxes, num=4, axis=-1)

    zeros = tf.zeros_like(t_y1)
    p_width = tf.maximum(zeros, p_x2 - p_x1)
    p_height = tf.maximum(zeros, p_y2 - p_y1)
    t_width = tf.maximum(zeros, t_x2 - t_x1)
    t_height = tf.maximum(zeros, t_y2 - t_y1)
    p_area = p_width * p_height
    t_area = t_width * t_height

    # intersection
    i_x1 = tf.maximum(t_x1, p_x1)
    i_y1 = tf.maximum(t_y1, p_y1)
    i_x2 = tf.minimum(t_x2, p_x2)
    i_y2 = tf.minimum(t_y2, p_y2)
    i_width = tf.maximum(zeros, i_x2 - i_x1)
    i_height = tf.maximum(zeros, i_y2 - i_y1)
    i_area = i_width * i_height

    # union
    u_area = p_area + t_area - i_area
    iou_v = tf.math.divide_no_nan(i_area, u_area)
    if iou_type == "iou":
        return iou_v
    
    # enclose 
    e_y1 = tf.minimum(p_y1, t_y1)
    e_x1 = tf.minimum(p_x1, t_x1)
    e_y2 = tf.maximum(p_y2, t_y2)
    e_x2 = tf.maximum(p_x2, t_x2)

    assert iou_type in ["diou", "ciou", "giou"]
    if iou_type == "giou":
        e_width = e_x2 - e_x1
        e_height = e_y2 - e_y1
        e_area = e_width * e_height
        giou_v = iou_v - tf.math.divide_no_nan(e_area - iou_v, e_area)

        return giou_v
    
    assert iou_type in ["diou", "ciou"]
    # box center
    p_center = tf.stack([(p_y1 + p_y2) / 2, (p_x1 + p_x2) / 2], axis=-1)
    t_center = tf.stack([(t_y1 + t_y2) / 2, (t_x1 + t_x2) / 2], axis=-1)

    center_dist = tf.linalg.norm(p_center - t_center, axis=-1) ** 2
    diag_dist = tf.linalg.norm(tf.stack([e_y2 - e_y1, e_x2 - e_x1], -1), axis=-1) ** 2
    diou_v = iou_v - tf.math.divide_no_nan(center_dist, diag_dist)

    if iou_type == "diou":
        return diou_v
    
    assert iou_type == "ciou"

    v = _get_v(p_height, p_width, t_height, t_width)
    alpha = tf.math.divide_no_nan(v, (1 - iou_v) + v)

    return diou_v - alpha * v


def compute_unaligned_iou(target_boxes, predicted_boxes, iou_type="iou"):
    """Computing the IoU for unaligned boxes.
    
    Args:
        predicted_boxes: predicted boxes, with coordinate [y_min, x_min, y_max, x_max].
        target_boxes: target boxes, with coordinate [y_min, x_min, y_max, x_max].
        iou_type: one of ['iou', 'ciou', 'diou', 'giou'].
    Returns:
        IoU loss float `Tensor`.
    """

    predicted_boxes = tf.expand_dims(predicted_boxes, 1)
    target_boxes = tf.expand_dims(target_boxes, 0)
    
    return compute_iou(target_boxes, predicted_boxes)
