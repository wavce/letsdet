import numpy as np
import tensorflow as tf


BBOX_XFORM_CLIP = np.log(1000. / 16.)


def bbox_overlap(boxes, gt_boxes):
    """Calculates the overlap between proposal and ground truth boxes.

    Some `gt_boxes` may have been padded.  The returned `iou` tensor for these
    boxes will be -1.

    Args:
        boxes: a tensor with a shape of [N, 4]. N is the number of
            proposals before groundtruth assignment (e.g., rpn_post_nms_topn). The
            last dimension is the pixel coordinates in [ymin, xmin, ymax, xmax] form.
        gt_boxes: a tensor with a shape of [MAX_NUM_INSTANCES, 4]. This
            tensor might have paddings with a negative value.
    Returns:
        iou: a tensor with as a shape of [N, MAX_NUM_INSTANCES].
    """
    with tf.name_scope("bbox_overlap"):
        bb_y_min, bb_x_min, bb_y_max, bb_x_max = tf.split(
            value=boxes, num_or_size_splits=4, axis=-1)
        gt_y_min, gt_x_min, gt_y_max, gt_x_max = tf.split(
            value=gt_boxes, num_or_size_splits=4, axis=-1)

        # Calculates the intersection area.
        i_xmin = tf.maximum(bb_x_min, tf.transpose(gt_x_min, [1, 0]))
        i_xmax = tf.minimum(bb_x_max, tf.transpose(gt_x_max, [1, 0]))
        i_ymin = tf.maximum(bb_y_min, tf.transpose(gt_y_min, [1, 0]))
        i_ymax = tf.minimum(bb_y_max, tf.transpose(gt_y_max, [1, 0]))
        i_area = tf.maximum((i_xmax - i_xmin), 0) * tf.maximum((i_ymax - i_ymin), 0)

        # Calculates the union area.
        bb_area = (bb_y_max - bb_y_min) * (bb_x_max - bb_x_min)
        gt_area = (gt_y_max - gt_y_min) * (gt_x_max - gt_x_min)
        # Adds a small epsilon to avoid divide-by-zero.
        u_area = bb_area + tf.transpose(gt_area, [1, 0]) - i_area + 1e-8

        # Calculates IoU.
        iou = i_area / u_area

        # Fills -1 for padded ground truth boxes.
        padding_mask = tf.less(i_xmin, tf.zeros_like(i_xmin))
        iou = tf.where(padding_mask, -tf.ones_like(iou), iou)

        return iou


def encode_boxes(boxes, anchors, weights=None):
    """Encode boxes to targets.

    Args:
        boxes: a tensor whose last dimension is 4 representing the coordinates
            of boxes in ymin, xmin, ymax, xmax order.
        anchors: a tensor whose shape is the same as `boxes` representing the
            coordinates of anchors in ymin, xmin, ymax, xmax order.
        weights: None or a list of four float numbers used to scale coordinates.

    Returns:
        encoded_boxes: a tensor whose shape is the same as `boxes` representing the
            encoded box targets.
    """
    with tf.name_scope('encode_box'):
        boxes = tf.cast(boxes, dtype=anchors.dtype)
        y_min = boxes[..., 0:1]
        x_min = boxes[..., 1:2]
        y_max = boxes[..., 2:3]
        x_max = boxes[..., 3:4]
        box_h = y_max - y_min + 1.0
        box_w = x_max - x_min + 1.0
        box_yc = y_min + 0.5 * box_h
        box_xc = x_min + 0.5 * box_w

        anchor_ymin = anchors[..., 0:1]
        anchor_xmin = anchors[..., 1:2]
        anchor_ymax = anchors[..., 2:3]
        anchor_xmax = anchors[..., 3:4]
        anchor_h = anchor_ymax - anchor_ymin + 1.0
        anchor_w = anchor_xmax - anchor_xmin + 1.0
        anchor_yc = anchor_ymin + 0.5 * anchor_h
        anchor_xc = anchor_xmin + 0.5 * anchor_w

        encoded_dy = (box_yc - anchor_yc) / anchor_h
        encoded_dx = (box_xc - anchor_xc) / anchor_w
        encoded_dh = tf.math.log(box_h / anchor_h)
        encoded_dw = tf.math.log(box_w / anchor_w)

        if weights:
            encoded_dy *= weights[0]
            encoded_dx *= weights[1]
            encoded_dh *= weights[2]
            encoded_dw *= weights[3]

        encoded_boxes = tf.concat(
            [encoded_dy, encoded_dx, encoded_dh, encoded_dw], axis=-1)

        return encoded_boxes


def decode_boxes(encoded_boxes, anchors, weights=None):
    """Decode boxes.

    Args:
        encoded_boxes: a tensor whose last dimension is 4 representing the
            coordinates of encoded boxes in ymin, xmin, ymax, xmax order.
        anchors: a tensor whose shape is the same as `boxes` representing the
            coordinates of anchors in ymin, xmin, ymax, xmax order.
        weights: None or a list of four float numbers used to scale coordinates.

    Returns:
        encoded_boxes: a tensor whose shape is the same as `boxes` representing the
            decoded box targets.
    """
    with tf.name_scope('decode_box'):
        encoded_boxes = tf.cast(encoded_boxes, dtype=anchors.dtype)
        dy = encoded_boxes[..., 0:1]
        dx = encoded_boxes[..., 1:2]
        dh = encoded_boxes[..., 2:3]
        dw = encoded_boxes[..., 3:4]
        if weights:
            dy /= weights[0]
            dx /= weights[1]
            dh /= weights[2]
            dw /= weights[3]
        
        dh = tf.minimum(dh, BBOX_XFORM_CLIP)
        dw = tf.minimum(dw, BBOX_XFORM_CLIP)

        anchor_ymin = anchors[..., 0:1]
        anchor_xmin = anchors[..., 1:2]
        anchor_ymax = anchors[..., 2:3]
        anchor_xmax = anchors[..., 3:4]

        anchor_h = anchor_ymax - anchor_ymin + 1.0
        anchor_w = anchor_xmax - anchor_xmin + 1.0
        anchor_yc = anchor_ymin + 0.5 * anchor_h
        anchor_xc = anchor_xmin + 0.5 * anchor_w

        decoded_boxes_yc = dy * anchor_h + anchor_yc
        decoded_boxes_xc = dx * anchor_w + anchor_xc
        decoded_boxes_h = tf.math.exp(dh) * anchor_h
        decoded_boxes_w = tf.math.exp(dw) * anchor_w

        decoded_boxes_ymin = decoded_boxes_yc - 0.5 * decoded_boxes_h
        decoded_boxes_xmin = decoded_boxes_xc - 0.5 * decoded_boxes_w
        decoded_boxes_ymax = decoded_boxes_ymin + decoded_boxes_h
        decoded_boxes_xmax = decoded_boxes_xmin + decoded_boxes_w
        
        decoded_boxes = tf.concat(
            [decoded_boxes_ymin, decoded_boxes_xmin,
            decoded_boxes_ymax, decoded_boxes_xmax],
            axis=-1)
        return decoded_boxes


def clip_boxes(boxes, height, width):
    """Clip boxes.

    Args:
        boxes: a tensor whose last dimension is 4 representing the coordinates
            of boxes in ymin, xmin, ymax, xmax order.
        height: an integer, a scalar or a tensor such as all but the last dimensions
            are the same as `boxes`. The last dimension is 1. It represents the height
            of the image.
        width: an integer, a scalar or a tensor such as all but the last dimensions
            are the same as `boxes`. The last dimension is 1. It represents the width
            of the image.

    Returns:
        clipped_boxes: a tensor whose shape is the same as `boxes` representing the
            clipped boxes.
    """
    with tf.name_scope('clip_box'):
        y_min = boxes[..., 0:1]
        x_min = boxes[..., 1:2]
        y_max = boxes[..., 2:3]
        x_max = boxes[..., 3:4]

        height = tf.cast(height, dtype=boxes.dtype)
        width = tf.cast(width, dtype=boxes.dtype)
        clipped_y_min = tf.maximum(tf.minimum(y_min, height - 1.0), 0.0)
        clipped_y_max = tf.maximum(tf.minimum(y_max, height - 1.0), 0.0)
        clipped_x_min = tf.maximum(tf.minimum(x_min, width - 1.0), 0.0)
        clipped_x_max = tf.maximum(tf.minimum(x_max, width - 1.0), 0.0)

        clipped_boxes = tf.concat(
            [clipped_y_min, clipped_x_min, clipped_y_max, clipped_x_max], axis=-1)
        return clipped_boxes


def filter_boxes(boxes, scores, min_size, height, width):
    """Filter out boxes that are too small.

    Args:
        boxes: a tensor whose last dimension is 4 representing the coordinates
            of boxes in ymin, xmin, ymax, xmax order.
        scores: a tensor such as all but the last dimensions are the same as
            `boxes`. The last dimension is 1. It represents the scores.
        min_size: an integer specifying the minimal size.
        height: an integer, a scalar or a tensor such as all but the last dimensions
            are the same as `boxes`. The last dimension is 1. It represents the height
            of the image.
        width: an integer, a scalar or a tensor such as all but the last dimensions
            are the same as `boxes`. The last dimension is 1. It represents the width
            of the image.

    Returns:
        filtered_boxes: a tensor whose shape is the same as `boxes` representing the
            filtered boxes.
        filtered_scores: a tensor whose shape is the same as `scores` representing
            the filtered scores.
    """
    with tf.name_scope('filter_box'):
        y_min = boxes[..., 0:1]
        x_min = boxes[..., 1:2]
        y_max = boxes[..., 2:3]
        x_max = boxes[..., 3:4]

        h = y_max - y_min + 1.0
        w = x_max - x_min + 1.0
        yc = y_min + h / 2.0
        xc = x_min + w / 2.0

        height = tf.cast(height, dtype=boxes.dtype)
        width = tf.cast(width, dtype=boxes.dtype)
     
        min_size = tf.cast(tf.maximum(min_size, 1), dtype=boxes.dtype)
        size_mask = tf.logical_and(tf.greater_equal(h, min_size), tf.greater_equal(w, min_size))
        center_mask = tf.logical_and(tf.less(yc, height), tf.less(xc, width))
        selected_mask = tf.logical_and(size_mask, center_mask)

        filtered_scores = tf.where(selected_mask, scores, tf.zeros_like(scores))
        filtered_boxes = tf.cast(selected_mask, dtype=boxes.dtype) * boxes
        
        return filtered_boxes, filtered_scores


def to_normalized_coordinates(boxes, height, width):
    """Converted absolute box coordinates to normalized ones.

    Args:
        boxes: a tensor whose last dimension is 4 representing the coordinates
            of boxes in ymin, xmin, ymax, xmax order.
        height: an integer, a scalar or a tensor such as all but the last dimensions
            are the same as `boxes`. The last dimension is 1. It represents the height
            of the image.
        width: an integer, a scalar or a tensor such as all but the last dimensions
            are the same as `boxes`. The last dimension is 1. It represents the width
            of the image.

    Returns:
        normalized_boxes: a tensor whose shape is the same as `boxes` representing
        the boxes in normalized coordinates.
    """
    with tf.name_scope('normalize_box'):
        height = tf.cast(height, dtype=boxes.dtype)
        width = tf.cast(width, dtype=boxes.dtype)

        y_min = boxes[..., 0:1] / height
        x_min = boxes[..., 1:2] / width
        y_max = boxes[..., 2:3] / height
        x_max = boxes[..., 3:4] / width

        normalized_boxes = tf.concat([y_min, x_min, y_max, x_max], axis=-1)
        normalized_boxes = tf.clip_by_value(normalized_boxes, 0, 1)
        
        return normalized_boxes


def to_absolute_coordinates(boxes, height, width):
    """Converted normalized box coordinates to absolute ones.

    Args:
        boxes: a tensor whose last dimension is 4 representing the coordinates
            of boxes in ymin, xmin, ymax, xmax order.
        height: an integer, a scalar or a tensor such as all but the last dimensions
            are the same as `boxes`. The last dimension is 1. It represents the height
            of the image.
        width: an integer, a scalar or a tensor such as all but the last dimensions
            are the same as `boxes`. The last dimension is 1. It represents the width
            of the image.

    Returns:
        absolute_boxes: a tensor whose shape is the same as `boxes` representing the
        boxes in absolute coordinates.
    """
    with tf.name_scope('denormalize_box'):
        height = tf.cast(height, dtype=boxes.dtype)
        width = tf.cast(width, dtype=boxes.dtype)

        y_min = boxes[..., 0:1] * height
        x_min = boxes[..., 1:2] * width
        y_max = boxes[..., 2:3] * height
        x_max = boxes[..., 3:4] * width

        absolute_boxes = tf.concat([y_min, x_min, y_max, x_max], axis=-1)
        return absolute_boxes
        