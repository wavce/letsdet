import tensorflow as tf


def _get_box_indices(batch_size, num_boxes):

    indices = tf.tile(tf.expand_dims(tf.range(batch_size), -1), [1, num_boxes])

    return tf.reshape(indices, [-1])


def _selective_crop_and_resize(features, boxes, box_levels, boundaries, output_height=7, output_width=7, feat_dims=256, sample_offset=0., align=True):
    # Compute the grid position w.r.t the corresponding feature map.
    with tf.name_scope("selective_crop_and_resize"):
        feat_shape = tf.shape(features)
        batch_size = feat_shape[0]
        max_feature_height = feat_shape[2]
        max_feature_width = feat_shape[3]
        num_boxes = tf.shape(boxes)[1]
        num_levels = tf.shape(features)[1]

        if align:
            boxes -= 0.5

        box_h = boxes[..., 2] - boxes[..., 0]
        box_w = boxes[..., 3] - boxes[..., 1]
        box_grid_yx = tf.TensorArray(size=output_height * output_width, dtype=boxes.dtype)
        bin_grid_h = box_h / output_height
        bin_grid_w = box_w / output_width
        
        for i in tf.range(output_height):
            gy = boxes[..., 0] + (tf.cast(i, boxes.dtype) + sample_offset) * bin_grid_h
            gy = tf.minimum(gy, boundaries[..., 0])
            for j in tf.range(output_width):
                gx = boxes[..., 1] + (tf.cast(j, boxes.dtype) + sample_offset) * bin_grid_w
                gx = tf.minimum(gx, boundaries[..., 1])
                box_grid_yx = box_grid_yx.write(i * output_width + j, tf.stack([gy, gx], -1))

        box_grid_yx = box_grid_yx.stack()
        box_grid_yx = tf.transpose(box_grid_yx, [1, 2, 0, 3])

        box_grid_y0x0 = tf.math.floor(box_grid_yx)
        box_grid_y1x1 = box_grid_y0x0 + 1.
        
        height_dim_offset = tf.cast(max_feature_width, box_grid_y0x0.dtype)
        level_dim_offset = tf.cast(max_feature_height * max_feature_width, box_grid_y0x0.dtype)
        batch_dim_offset = tf.cast(num_levels, box_grid_y0x0.dtype) * level_dim_offset

        batch_dim_inds = tf.reshape(
            tf.range(batch_size, dtype=box_grid_y0x0.dtype) * batch_dim_offset,
            [batch_size, 1, 1, 1])
        box_levels = tf.cast(tf.expand_dims(box_levels, -1), box_grid_y0x0.dtype)
        box_level_inds = tf.reshape(box_levels * level_dim_offset, [batch_size, num_boxes, 1, 1])
        batch_level_inds = batch_dim_inds + box_level_inds

        y0x0 = batch_level_inds + box_grid_y0x0[..., 0] * height_dim_offset + box_grid_y0x0[..., 1]
        y0x1 = batch_level_inds + box_grid_y0x0[..., 0] * height_dim_offset + box_grid_y1x1[..., 1]
        y1x0 = batch_level_inds + box_grid_y1x1[..., 0] * height_dim_offset + box_grid_y0x0[..., 1]
        y1x1 = batch_level_inds + box_grid_y1x1[..., 0] * height_dim_offset + box_grid_y1x1[..., 1]
    
        f00 = tf.gather(tf.reshape(features, [-1, feat_dims]), tf.cast(tf.reshape(y0x0, [-1]), tf.int32))
        f01 = tf.gather(tf.reshape(features, [-1, feat_dims]), tf.cast(tf.reshape(y0x1, [-1]), tf.int32))
        f10 = tf.gather(tf.reshape(features, [-1, feat_dims]), tf.cast(tf.reshape(y1x0, [-1]), tf.int32))
        f11 = tf.gather(tf.reshape(features, [-1, feat_dims]), tf.cast(tf.reshape(y1x1, [-1]), tf.int32))

        f00 = tf.reshape(f00, [batch_size, num_boxes, output_height, output_width, feat_dims])
        f01 = tf.reshape(f01, [batch_size, num_boxes, output_height, output_width, feat_dims])
        f10 = tf.reshape(f10, [batch_size, num_boxes, output_height, output_width, feat_dims])
        f11 = tf.reshape(f11, [batch_size, num_boxes, output_height, output_width, feat_dims])
       
        box_grid_yx = tf.reshape(box_grid_yx, [batch_size, num_boxes, output_height, output_width, 2])
        box_grid_y0x0 = tf.reshape(box_grid_y0x0, [batch_size, num_boxes, output_height, output_width, 2])

        # The RoIAlign feature f can be computed by bilinear interpolation of four
        # neighboring feature points f0, f1, f2, and f3.
        # f(y, x) = [hy, ly] * [[f00, f01], * [hx, lx]^T
        #                       [f10, f11]]
        # f(y, x) = (hy*hx)f00 + (hy*lx)f01 + (ly*hx)f10 + (lx*ly)f11
        # f(y, x) = w00*f00 + w01*f01 + w10*f10 + w11*f11
        lyx = box_grid_yx - box_grid_y0x0
        hyx = 1.0 - lyx
             
        w00 = hyx[..., 0:1] * hyx[..., 1:2]
        w01 = hyx[..., 0:1] * lyx[..., 1:2]
        w10 = hyx[..., 1:2] * lyx[..., 0:1]
        w11 = lyx[..., 0:1] * lyx[..., 1:2]
        features = tf.add_n([w00 * f00, w01 * f01, w10 * f10, w11 * f11])  

        return features


class MultiLevelAlignedRoIPooling(tf.keras.layers.Layer):
    """
    crop_size: A list of two integers `[crop_height, crop_width]`. All
      cropped image patches are resized to this size. The aspect ratio of the
      image content is not preserved. Both `crop_height` and `crop_width` need
      to be positive.
    """
    def __init__(self,
                 pooled_size=7,
                 feat_dims=256,
                 min_level=2,
                 max_level=5,
                 **kwargs):
        super(MultiLevelAlignedRoIPooling, self).__init__(**kwargs)

        self.pooled_size = pooled_size

        self.feat_dims = feat_dims
        self.min_level = min_level
        self.max_level = max_level

    def multi_levels_aligned_roi_pooling(self, features, boxes):
        """Crop and resize on multilevel feature pyramid.
           Generate the (output_size, output_size) set of pixels for each input box
           by first locating the box into the correct feature level, and then cropping
           and resizing it using the corresponding feature map of that level.

            Args:
                features: A list, The features are in shape of [batch_size, height, width, num_filters].
                boxes: A 3-D Tensor of shape [batch_size, num_boxes, 4]. Each row
                    represents a box with [y1, x1, y2, x2] in un-normalized coordinates.
            Returns:
                A 5-D tensor representing feature crop of shape
                [batch_size, num_boxes, output_size, output_size, num_filters].
        """
        with tf.name_scope("multi_levels_aligned_roi_pooling"):
            max_feature_height = tf.shape(features[0])[1]
            max_feature_width = tf.shape(features[0])[2]

            assert (
                len(features) == (self.max_level - self.min_level + 1)
            ), "unequal value, num_level_assignments={}, but x is list of {} features".format(
                (self.max_level - self.min_level + 1), len(features))

            features_all = tf.stack([
                tf.image.pad_to_bounding_box(feat,
                                             offset_height=0,
                                             offset_width=0,
                                             target_height=max_feature_height,
                                             target_width=max_feature_width)
                for feat in features], axis=1)
            
            features_all = tf.cast(features_all, tf.float32)
            boxes = tf.cast(boxes, tf.float32)

            # Assigns boxes to right level
            box_height = boxes[:, :, 2] - boxes[:, :, 0]
            box_width = boxes[:, :, 3] - boxes[:, :, 1]
            area_sqrt = tf.math.sqrt(box_height * box_width)
            levels = tf.cast(tf.math.floordiv(tf.math.log(tf.math.divide(area_sqrt, 224.0)),
                                              tf.math.log(2.0)) + 4.0, dtype=tf.int32)
       
            # Maps levels between [min_level, max_level]
            levels = tf.minimum(self.max_level, tf.maximum(levels, self.min_level))
            # Projects box location and sizes to corresponding feature levels.
            scale_to_level = tf.cast(
                tf.math.pow(tf.constant(2.0), tf.cast(levels, tf.float32)), dtype=boxes.dtype)
            
            rois = boxes / tf.expand_dims(scale_to_level, axis=2)
            # box_width /= scale_to_level
            # box_height /= scale_to_level
            # Maps levels to [0, max_level-min_level]
            levels -= self.min_level
            level_strides = tf.pow([[2.0]], tf.cast(levels, tf.float32))
            max_feature_height = tf.cast(max_feature_height, level_strides.dtype)
            max_feature_width = tf.cast(max_feature_width, level_strides.dtype)
            boundaries = tf.stack([max_feature_height / level_strides - 1, max_feature_width / level_strides - 1], -1)
            print(level_strides)
            roi_features = _selective_crop_and_resize(
                features_all, rois, levels, boundaries, self.pooled_size, self.pooled_size, self.feat_dims)
            
            return roi_features

    def call(self, inputs, proposals):
        return self.multi_levels_aligned_roi_pooling(inputs, proposals)

    def get_config(self):
        config = {"pooled_size": self.pooled_size}

        base_config = super(MultiLevelAlignedRoIPooling, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))


class SingleLevelAlignedRoIPooling(tf.keras.layers.Layer):
    def __init__(self, pooled_size=7, strides=16, **kwargs):
        super(SingleLevelAlignedRoIPooling, self).__init__(**kwargs)

        self.pooled_size = pooled_size
        self.strides = strides
    
    def single_level_aligned_roi_pooling(self, features, boxes):
        """
        Args:
            features: A `Tensor`. Must be one of the following types: `uint8`, `int8`,
            `int16`, `int32`, `int64`, `half`, 'bfloat16', `float32`, `float64`.
            A 4-D tensor of shape `[batch, height, width, depth]`.
    
            boxes: A `Tensor` of type `float32`, 'bfloat16' or `float16`.
                 A 3-D tensor of shape `[batch, num_boxes, 4]`. The boxes are specified in
                normalized coordinates and are of the form `[y1, x1, y2, x2]`. A
                normalized coordinate value of `y` is mapped to the image coordinate at
                `y * (image_height - 1)`, so as the `[0, 1]` interval of normalized image
                height is mapped to `[0, image_height - 1] in image height coordinates.
                The width dimension is treated similarly.
    
        Returns:
            pooled_features: A 5-D tensor of shape `[batch, num_boxes, crop_height, crop_width, depth]` 
        """
        batch_size = tf.shape(features)[0]
        depth = tf.shape(features)[3]
        num_boxes = tf.shape(boxes)[1]

        boxes = tf.reshape(boxes, [-1, 4])

        pooled_features = tf.image.crop_and_resize(image=features,
                                                   boxes=boxes,
                                                   box_indices=_get_box_indices(batch_size, num_boxes),
                                                   crop_size=[self.pooled_size, self.pooled_size])
        pooled_features = tf.reshape(pooled_features, [batch_size, num_boxes, self.pooled_size, self.pooled_size, depth])
        
        return pooled_features
    
    def call(self, inputs, proposals):
        return self.single_level_aligned_roi_pooling(inputs, proposals)

    def get_config(self):
        config = {"pooled_size": self.pooled_size}

        base_config = super(SingleLevelAlignedRoIPooling, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))


def main():
    import torch
    import numpy as np
    import random
    from detectron2.structures import Boxes
    from detectron2.modeling.poolers import ROIPooler
    # from detectron2.layers import ROIAlign
    from torchvision.ops import roi_align

    # np.printoptions(precision=4)
    # img = np.arange(25).reshape(5, 5).astype("float32")
    # img = np.tile(np.expand_dims(img, -1), [1, 1, 3])
    # inputs = tf.convert_to_tensor(img)
    # inputs = tf.reshape(inputs, [1, 1, 5, 5, 3])
    # boxes = tf.constant([[[1, 1, 3, 3]]], dtype=tf.float32)
    # pooled = _selective_crop_and_resize(inputs, boxes, tf.constant([[0]]), tf.constant([[5, 5]], tf.float32), 3, 3, 3, 0.5, True)
    # print(pooled[0, 0, :, :, 0])
    # inputs = torch.from_numpy(img.transpose(2, 0, 1)[None, :, :].astype("float32"))
    # rois = torch.from_numpy(np.array([0, 1, 1, 3, 3]).astype("float32"))[None, :]
    # output = roi_align(inputs, rois, (3, 3), 1, 0, True)
    # print(output[0, 0])

    np.random.seed(4)
    noise = np.random.uniform(0, 1, [32, 32, 1])
    img = np.arange(32 * 32).reshape(32, 32, 1).astype("float32")
    img += noise
    inputs = tf.convert_to_tensor(img, tf.float32)
    inputs = tf.reshape(inputs, [1, 1, 32, 32, 1])
    boxes = tf.constant([[[1, 1, 17, 17]]], dtype=tf.float32)
    pooled = _selective_crop_and_resize(inputs, boxes, tf.constant([[0]]), tf.constant([[31, 31]], tf.float32), 5, 5, 1, 0.5, True)
    
    print(pooled[0, 0, :, :, 0])
    inputs = torch.from_numpy(img.transpose(2, 0, 1)[None, :, :].astype("float32"))
    # print(inputs.shape)
    rois = torch.from_numpy(np.array([0, 1, 1, 17, 17]).astype("float32"))[None, :]
    output = roi_align(inputs, rois, (5, 5), 1, 0, True)
    output = output.permute(0, 2, 3, 1)
    print(output[0, :, :, 0])


    # feat_dims = 1
    # pooling = MultiLevelAlignedRoIPooling(5, feat_dims, 2, 5)
    # features = [tf.random.uniform([1, 1024 // s, 1024 // s, feat_dims]) for s in [4, 8, 16, 32]]
    # yx = tf.random.uniform([1, 1, 2], 0, 1) * 1024
    # hw = tf.random.uniform([1, 1, 2]) * 1024
    # boxes = tf.clip_by_value(tf.concat([yx - hw * 0.5, yx + hw * 0.5], -1), 0, 1024)
    # pooled_features = pooling(features, boxes)
    # print(pooled_features[0, 0, ..., 0].numpy())

    # pooler = ROIPooler((5, 5), (0.25, 0.125, 0.0625, 0.03125), 0, "ROIAlignV2")
    # features = [torch.from_numpy(f.numpy()).permute(0, 3, 1, 2) for f in features]
    # boxes = [Boxes(torch.from_numpy(np.stack([b[..., 1], b[..., 0], b[..., 3], b[..., 2]], -1))) for b in boxes.numpy()]
    # f2 = pooler(features, boxes)
    # print(f2[0, 0].numpy())
   
if __name__ == '__main__':
    main()
