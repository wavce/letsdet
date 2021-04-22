import tensorflow as tf
from .assigner import Assigner
from ..builder import ASSIGNERS

INF = 10e10


@ASSIGNERS.register
class FCOSAssigner(Assigner):
    def __init__(self, sampling_radius=.0, epsilon=1e-4, **kwargs):
        super(FCOSAssigner, self).__init__(Assigner)

        self.sampling_radius = sampling_radius
        self.epsilon = epsilon

    def get_sample_region(self, gt_boxes, grid_y, grid_x, strides):
        with tf.name_scope("get_sample_region"):
            gt_boxes = tf.tile(gt_boxes, [tf.shape(grid_x)[0], 1, 1])
            center_y = (gt_boxes[..., 0] + gt_boxes[..., 2]) * 0.5
            center_x = (gt_boxes[..., 1] + gt_boxes[..., 3]) * 0.5

            y_min = center_y - strides * self.sampling_radius
            x_min = center_x - strides * self.sampling_radius
            y_max = center_y + strides * self.sampling_radius
            x_max = center_x + strides * self.sampling_radius

            center_gt_y_min = tf.where(y_min > gt_boxes[..., 0], y_min, gt_boxes[..., 0])
            center_gt_x_min = tf.where(x_min > gt_boxes[..., 1], x_min, gt_boxes[..., 1])
            center_gt_y_max = tf.where(y_max > gt_boxes[..., 2], gt_boxes[..., 2], y_max)
            center_gt_x_max = tf.where(x_max > gt_boxes[..., 3], gt_boxes[..., 3], x_max)

            top = grid_y[:, None] - center_gt_y_min
            left = grid_x[:, None] - center_gt_x_min
            bottom = center_gt_y_max - grid_y[:, None]
            right = center_gt_x_max - grid_x[:, None]
            center_box = tf.stack([top, left, bottom, right], -1)

            return tf.greater(tf.reduce_min(center_box, -1), 0.01)

    def assign(self, gt_boxes, gt_labels, grid_y, grid_x, strides, object_size_of_interest):
        with tf.name_scope("assigner"):            
            h, w = tf.shape(grid_x)[0], tf.shape(grid_y)[1]
            num_grid = h * w   # h * w
            grid_y = tf.reshape(grid_y, [num_grid])
            grid_x = tf.reshape(grid_x, [num_grid])

            valid_mask = tf.logical_not(tf.reduce_all(gt_boxes == 0, 1))
            gt_boxes = tf.boolean_mask(gt_boxes, valid_mask)
            gt_boxes = tf.concat([tf.zeros([1, 4], gt_boxes.dtype), gt_boxes], 0)
            gt_labels = tf.concat([tf.zeros([1], gt_labels.dtype), gt_labels], 0)
            # normalizer = tf.cast([h * strides, w * strides, h * strides, w * strides], gt_boxes.dtype)
            # gt_boxes *= normalizer

            gt_boxes = tf.expand_dims(gt_boxes, 0)  # (1, n, 4)
            gt_areas = (gt_boxes[..., 2] - gt_boxes[..., 0]) * (gt_boxes[..., 3] - gt_boxes[..., 1])  # (1, n)
            distances = tf.stack([grid_y[:, None] - gt_boxes[..., 0],
                                  grid_x[:, None] - gt_boxes[..., 1],
                                  gt_boxes[..., 2] - grid_y[:, None],
                                  gt_boxes[..., 3] - grid_x[:, None]], axis=2)  # (h * w, n, 4)

            if self.sampling_radius > 0:
                in_box_mask = self.get_sample_region(gt_boxes, grid_y, grid_x, strides)
            else:
                in_box_mask = tf.greater_equal(tf.reduce_min(distances, 2), 0)  # (h * w, n)

            max_distances = tf.reduce_max(distances, 2)
            in_level_mask = tf.logical_and(
                tf.greater_equal(max_distances, object_size_of_interest[0]),
                tf.less(max_distances, object_size_of_interest[1]))  # (h * w, n)
            
            gt_areas = tf.tile(gt_areas, [num_grid, 1])  # (h * w, n)
            
            mask = tf.logical_and(in_box_mask, in_level_mask)
            gt_areas = tf.where(mask, gt_areas, tf.ones_like(gt_areas) * INF)

            min_gt_area_indices = tf.argmin(gt_areas, 1)
            indices = tf.stack([tf.cast(tf.range(num_grid), min_gt_area_indices.dtype), min_gt_area_indices], axis=1)
            target_boxes = tf.gather_nd(tf.tile(gt_boxes, [num_grid, 1, 1]), indices)
            target_labels = tf.gather(gt_labels, min_gt_area_indices)

            target_centerness = self.compute_centerness(target_boxes, grid_y, grid_x)
            target_boxes = tf.reshape(target_boxes, [h * w, 4])
            target_centerness = tf.reshape(target_centerness, [h * w, 1])

            return target_boxes, target_labels, target_centerness

    def compute_centerness(self, target_boxes, grid_y, grid_x):
        with tf.name_scope("compute_centerness"):
            l_ = grid_x - target_boxes[:, 1]
            t_ = grid_y - target_boxes[:, 0]
            r_ = target_boxes[:, 3] - grid_x
            b_ = target_boxes[:, 2] - grid_y

            left_right = tf.stack([l_, r_], -1)
            top_bottom = tf.stack([t_, b_], -1)

            centerness = tf.math.sqrt(
                (tf.reduce_min(left_right, -1) / (tf.reduce_max(left_right, -1) + self.epsilon)) * 
                (tf.reduce_min(top_bottom, -1) / (tf.reduce_max(top_bottom, -1) + self.epsilon)))
           
            return centerness

    def __call__(self, gt_boxes, gt_labels, grid_y, grid_x, strides, object_size_of_interest):
        with tf.name_scope("fcos_assigner"):
            return self.assign(gt_boxes, gt_labels, grid_y, grid_x, strides, object_size_of_interest)
        
