import tensorflow as tf
from ..builder import AUGMENTATIONS 


AUGMENTATIONS.register
class Mosaic(object):
    """Moasic Augmentation."""
    def __init__(self, size, min_image_scale=0.25, prob=0.5, max_boxes=200):
        """
        Args:
            size: the model input size.
            min_image_scale: minimum percentage of out_size dimension
                should the mosaic be. i.e if size is (680,680) and
                min_image_scale is 0.25 , minimum mosaic sub images
                dimension will be 25 % of 680.
        """
        size = [size] * 2 if not isinstance(size, (tuple, list)) else size
        self.size = size
        self.min_image_scale = min_image_scale
        self.prob = prob
        self.max_boxes = max_boxes
        assert (min_image_scale > 0), "Minimum Mosaic image dimension should be above 0"
    
    def _clip_boxes(self, boxes, labels, min_y, min_x, max_y, max_x):
        with tf.name_scope("clip_boxes"):
            min_y = tf.cast(min_y, boxes.dtype)
            min_x = tf.cast(min_x, boxes.dtype)
            max_y = tf.cast(max_y, boxes.dtype)
            max_x = tf.cast(max_x, boxes.dtype)
            y1 = tf.clip_by_value(boxes[..., 0], min_y, max_y) - min_y
            x1 = tf.clip_by_value(boxes[..., 1], min_x, max_x) - min_x
            y2 = tf.clip_by_value(boxes[..., 2], min_y, max_y) - min_y
            x2 = tf.clip_by_value(boxes[..., 3], min_x, max_x) - min_x
            
            clipped_boxes = tf.stack([y1, x1, y2, x2], -1)
            valid_mask = tf.cast(tf.logical_and(y2 - y1 > 0, x2 - x1 > 0), boxes.dtype)
            clipped_boxes = clipped_boxes * tf.expand_dims(valid_mask, -1)
            clipped_labels = labels * tf.cast(valid_mask, labels.dtype)

            greater_mask = tf.logical_and(
                tf.greater((clipped_boxes[:, 2] - clipped_boxes[:, 0]), 8),
                tf.greater(clipped_boxes[:, 3] - clipped_boxes[:, 1], 8))
            float_greater = tf.cast(greater_mask, clipped_boxes.dtype)
            clipped_boxes *= tf.expand_dims(float_greater, -1)
            clipped_labels *= tf.cast(float_greater, clipped_labels.dtype)
                        
            return clipped_boxes, clipped_labels
    
    def _valid_image(self, image, valid_size):
        with tf.name_scope("resize_image"):
            int_valid_size = tf.cast(valid_size, tf.int32)
            image = image[0:int_valid_size[0], 0:int_valid_size[1]]

            image_shape = tf.shape(image)[0:2]
            float_image_shape = tf.cast(image_shape, tf.float32)
            scale = 640 / tf.reduce_min(float_image_shape)
            if tf.reduce_any(scale > 1.):
                new_size = tf.cast(float_image_shape * scale, tf.int32)
                new_image = tf.image.resize(image, new_size)
                image = tf.cast(new_image, image.dtype)

            return image
    
    def _scale_image(self, images, boxes, labels, valid_size):
        with tf.name_scope("scale_image"):
            x = tf.random.uniform(
                shape=[], 
                minval=tf.cast(self.size[1] * self.min_image_scale, tf.int32),
                maxval=tf.cast(self.size[1] * (1. - self.min_image_scale), tf.int32),
                dtype=tf.int32, name="x")
            y = tf.random.uniform(
                shape=[], 
                minval=tf.cast(self.size[0] * self.min_image_scale, tf.int32),
                maxval=tf.cast(self.size[0] * (1. - self.min_image_scale), tf.int32),
                dtype=tf.int32, name="y")
            
            valid_size = tf.cast(valid_size, tf.int32)
            tl_img = self._valid_image(images[0], valid_size[0])
            vs0 = tf.shape(tl_img)[0:2]
            tl_y1 = tf.random.uniform([], 0, vs0[0] - y - 1, dtype=tf.int32, name="tl_y1")
            tl_x1 = tf.random.uniform([], 0, vs0[1] - x - 1, dtype=tf.int32, name="tl_x1")
            topleft = tl_img[tl_y1:tl_y1 + y, tl_x1:tl_x1 + x]
            float_y = tf.cast(y, boxes[0].dtype)
            float_x = tf.cast(x, boxes[0].dtype)
            tl_boxes, tl_labels = self._clip_boxes(boxes[0], labels[0], tl_y1, tl_x1, tl_y1 + y, tl_x1 + x)
                        
            tr_img = self._valid_image(images[1], valid_size[1])
            vs1 = tf.shape(tr_img)[0:2]
            tr_y1 = tf.random.uniform([], 0, vs1[0] - y - 1, dtype=tf.int32, name="tr_y1")
            tr_x1 = tf.random.uniform([], 0, vs1[1] - (self.size[1] - x) - 1, dtype=tf.int32, name="tr_x1")
            topright = tr_img[tr_y1:tr_y1 + y, tr_x1:tr_x1 + (self.size[1] - x)]
            tr_boxes, tr_labels = self._clip_boxes(boxes[1], labels[1], tr_y1, tr_x1, tr_y1 + y, tr_x1 + (self.size[1] - x))
            tr_boxes += tf.stack([0, float_x, 0, float_x], 0)
            
            bl_img = self._valid_image(images[2], valid_size[2])
            vs2 = tf.shape(bl_img)[0:2]
            bl_y1 = tf.random.uniform([], 0, vs2[0] - (self.size[0] - y) - 1, dtype=tf.int32, name="bl_y1")
            bl_x1 = tf.random.uniform([], 0, vs2[1] - x - 1, dtype=tf.int32, name="bl_x1")
            bottomleft = bl_img[bl_y1:bl_y1 + (self.size[0] - y), bl_x1:bl_x1 + x]
            bl_boxes, bl_labels = self._clip_boxes(boxes[2], labels[2], bl_y1, bl_x1, bl_y1 + (self.size[0] - y), bl_x1 + x)
            bl_boxes += tf.stack([float_y, 0, float_y, 0], 0)
            
            br_img = self._valid_image(images[3], valid_size[3])
            vs3 = tf.shape(br_img)[0:2]
            br_y1 = tf.random.uniform([], 0, vs3[0] - (self.size[0] - y) - 1, dtype=tf.int32, name="br_y1")
            br_x1 = tf.random.uniform([], 0, vs3[1] - (self.size[1] - x) - 1, dtype=tf.int32, name="br_x1")
            bottomright = br_img[br_y1:br_y1 + (self.size[0] - y), br_x1:br_x1 + (self.size[1] - x)]
            br_boxes, br_labels = self._clip_boxes(boxes[3], labels[3], br_y1, br_x1, br_y1 + (self.size[0] - y), br_x1 + (self.size[1] - x))
            br_boxes += tf.stack([float_y, float_x, float_y, float_x], 0)
            
            top = tf.concat([topleft, topright], 1, name="top")
            bottom = tf.concat([bottomleft, bottomright], 1, name="bottom")
            mosaic = tf.concat([top, bottom], 0, name="mosaic")
            mosaic = tf.cast(mosaic, images[0].dtype)

            mosaic_boxes = tf.concat([tl_boxes, tr_boxes, bl_boxes, br_boxes], 0, name="mosaic_boxes")
            mosaic_labels = tf.concat([tl_labels, tr_labels, bl_labels, br_labels], 0, name="mosaic_labels")

            is_zeros = tf.not_equal(mosaic_labels, 0)
            mosaic_boxes = tf.boolean_mask(mosaic_boxes, is_zeros)
            mosaic_labels = tf.boolean_mask(mosaic_labels, is_zeros)
            num = tf.size(mosaic_labels)
            if num < self.max_boxes:
                mosaic_labels = tf.concat([mosaic_labels, tf.zeros([self.max_boxes - num], mosaic_labels.dtype)], 0)
                mosaic_boxes = tf.concat([mosaic_boxes, tf.zeros([self.max_boxes - num, 4], mosaic_boxes.dtype)], 0)
            else:
                mosaic_boxes = mosaic_boxes[:self.max_boxes]
                mosaic_labels = mosaic_labels[:self.max_boxes]

            mosaic_valid_size = tf.constant(self.size, dtype=tf.float32)
            mosaic = tf.cast(mosaic, tf.uint8)
            
            return mosaic, mosaic_boxes, mosaic_labels, mosaic_valid_size
        
    def __call__(self, images, images_info):
        """Builds mosaic with given images, boxes, labels."""
        with tf.name_scope("mosaic"):
            shape = tf.shape(images)
            num_assert = tf.assert_equal(shape[0] % 4, 0, message="The number of image shoule divide by 4")
            
            with tf.control_dependencies([num_assert]):
                reshaped_images = tf.reshape(images, [shape[0] // 4, 4, shape[1], shape[2], shape[3]])
                boxes = tf.reshape(images_info["boxes"], [shape[0] // 4, 4, -1, 4])
                labels = tf.reshape(images_info["labels"], [shape[0] // 4, 4, -1])
                valid_size = tf.reshape(images_info["valid_size"], [shape[0] // 4, 4, -1])
            
            def _true_fn(inp):
                return self._scale_image(*inp)
            
            def _false_fn(inp):
                # i = tf.random.uniform([], 0, 4, tf.int32)
                img = inp[0][:, 0]
                b = inp[1][:, 0]
                l = inp[2][:, 0]
                vs = inp[3][:, 0]
                img = img[0:self.size[0], 0:self.size[1]]
                b, l = self._clip_boxes(b, l, 0, 0, self.size[0], self.size[1])
                vs = tf.clip_by_value(vs, 0, self.size[0])

                return img, b, l, vs
            
            mosaic, mosaic_boxes, mosaic_labels, mosaic_valid_size = tf.cond(
                tf.random.uniform([]) >= (1. - self.prob),
                lambda: tf.map_fn(_true_fn, (reshaped_images, boxes, labels, valid_size)),
                lambda: tf.map_fn(_false_fn, (reshaped_images, boxes, labels, valid_size)))

            images_info["input_size"] = mosaic_valid_size
            images_info["valid_size"] = mosaic_valid_size
            images_info["boxes"] = mosaic_boxes
            images_info["labels"] = mosaic_labels
            
            return mosaic, images_info
            # return reshaped_images, images_info
            
            
            
        
