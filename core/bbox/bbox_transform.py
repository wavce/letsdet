import tensorflow as tf
from utils import box_utils


class Box2Delta(object):
    def __init__(self, weights=None):
        self.weights = weights

    def __call__(self, proposals, boxes):
        return box_utils.encode_boxes(boxes, proposals, self.weights)


class Delta2Box(object):
    def __init__(self, weights=None):
        self.weights = weights

    def __call__(self, proposals, delta):
        return box_utils.decode_boxes(delta, proposals, self.weights)


class Distance2Box(object):
    def __call__(self, distances, grid_y, grid_x):
        with tf.name_scope("distance2box"):
            grid_y = tf.cast(tf.expand_dims(grid_y, 0), distances.dtype)
            grid_x = tf.cast(tf.expand_dims(grid_x, 0), distances.dtype)

            boxes = tf.stack([grid_y - distances[..., 0],
                              grid_x - distances[..., 1],
                              grid_y + distances[..., 2],
                              grid_x + distances[..., 3]], axis=-1)
            
            return boxes


class Box2Distance(object):
    def __call__(self, boxes, grid_y, grid_x):
        with tf.name_scope("box2distance"):
            grid_y = tf.cast(tf.expand_dims(grid_y, 0), boxes.dtype)
            grid_x = tf.cast(tf.expand_dims(grid_x, 0), boxes.dtype)

            dist = tf.stack([grid_y - boxes[..., 0],
                             grid_x - boxes[..., 1],
                             boxes[..., 2] - grid_y,
                             boxes[..., 3] - grid_x], axis=-1)
            
            return dist
