import math
import numpy as np
import tensorflow as tf
from ..builder import AUGMENTATIONS


def should_apply_op(prob):
    return tf.cast(tf.floor(tf.random.uniform([], dtype=tf.float32) + prob), tf.bool)


def jaccard(boxes, patch):
    lt = tf.maximum(boxes[:, 0:2], patch[0:2])
    rb = tf.minimum(boxes[:, 2:4], patch[2:4])

    wh = tf.maximum(0.0, rb - lt)  # (n, m, 2)
    overlap = tf.reduce_prod(wh, axis=1)  # (n, m)
    box_areas = tf.reduce_prod(boxes[:, 2:4] - boxes[:, 0:2], axis=1)  # (n, m)
    patch_area = (patch[2] - patch[0]) * (patch[3] - patch[1])

    overlaps = overlap / (box_areas + patch_area - overlap)

    return overlaps


def is_box_center_in_patch(boxes, patch):
    box_ctr = (boxes[:, 0:2] + boxes[:, 2:4]) * 0.5
    flags = tf.logical_and(x=tf.greater(box_ctr, patch[0:2]),
                           y=tf.less(box_ctr, patch[2:4]))
    flags = tf.reduce_all(flags, axis=1)

    return flags


def clip_boxes(boxes, labels, patch, base_center=True):
    if base_center:
        mask = is_box_center_in_patch(boxes, patch)
    else:
        mask = tf.logical_and(tf.greater(boxes[:, 2], boxes[:, 0] + 5),
                              tf.greater(boxes[:, 3], boxes[:, 1] + 5))

    clipped_boxes = tf.boolean_mask(boxes, mask)
    clipped_labels = tf.boolean_mask(labels, mask)
    

    patch_xy = tf.convert_to_tensor([patch[0], patch[1]] * 2, dtype=boxes.dtype)
    patch_wh = tf.convert_to_tensor([patch[2] - patch[0], patch[3] - patch[1]] * 2, boxes.dtype)
    clipped_boxes -= patch_xy
    clipped_boxes = tf.stack([tf.clip_by_value(clipped_boxes[:, 0], 0, patch_wh[0]),
                              tf.clip_by_value(clipped_boxes[:, 1], 0, patch_wh[1]),
                              tf.clip_by_value(clipped_boxes[:, 2], 0, patch_wh[0]),
                              tf.clip_by_value(clipped_boxes[:, 3], 0, patch_wh[1])], axis=-1)

    return clipped_boxes, clipped_labels


def resize_with_keep_ratio(image, scale, method=tf.image.ResizeMethod.BILINEAR):
    with tf.name_scope("resize_with_keep_ratio"):
        img_hw = tf.cast(tf.shape(image)[0:2], tf.float32)
        scale_factor = tf.constant(1.)
        scale = tf.cast(scale, tf.float32)
        if tf.size(scale) == 1:
            scale_factor = scale
        if tf.size(scale) == 2:
            long_side = tf.reduce_max(scale)
            short_side = tf.reduce_min(scale)
            scale_factor = tf.minimum(long_side / tf.reduce_max(img_hw), short_side / tf.reduce_min(img_hw))
        new_size = tf.stack([img_hw[0] * scale_factor + 0.5, img_hw[1] * scale_factor + 0.5])
        new_size = tf.cast(new_size, tf.int32)
        new_img = tf.image.resize(image, new_size, method=method)

        return new_img


@AUGMENTATIONS.register
class Resize(object):
    """Resize images & bbox & mask.

        This transform resizes the input image to some scale. Bboxes and masks are
        then resized with the same scale factor. If the input dict contains the key
        "scale", then the scale in the input dict is used, otherwise the specified
        scale in the init method is used. If the input dict contains the key
        "scale_factor" (if MultiScaleFlipAug does not give img_scale but
        scale_factor), the actual scale will be computed by image shape and
        scale_factor.

        `img_scale` can either be a tuple (single-scale) or a list of tuple
        (multi-scale). There are 3 multiscale modes:

        - ``ratio_range is not None``: randomly sample a ratio from the ratio \
        range and multiply it with the image scale.
        - ``ratio_range is None`` and ``multiscale_mode == "range"``: randomly \
        sample a scale from the multiscale range.
        - ``ratio_range is None`` and ``multiscale_mode == "value"``: randomly \
        sample a scale from multiple scales.

        Args:
            img_scale (tuple or list[tuple]): Images scales for resizing.
            multiscale_mode (str): Either "range" or "value".
            ratio_range (tuple[float]): (min_ratio, max_ratio)
            keep_ratio (bool): Whether to keep the aspect ratio when resizing the
                image.
            bbox_clip_border (bool, optional): Whether clip the objects outside
                the border of the image. Defaults to True.
            override (bool, optional): Whether to override `scale` and
                `scale_factor` so as to call resize twice. Default False. If True,
                after the first resizing, the existed `scale` and `scale_factor`
                will be ignored so the second resizing can be allowed.
                This option is a work-around for multiple times of resize in DETR.
                Defaults to False.
    """
    def __init__(self,
                 img_scale=None,
                 multiscale_mode='range',
                 ratio_range=None,
                 keep_ratio=True,
                 bbox_clip_border=True,
                 override=False):
        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, (tuple, list)):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]

        if ratio_range is not None:
            # mode 1: given a scale and a range of image ratio
            assert len(self.img_scale) == 1
        else:
            # mode 2: given multiple scales or a range of scales
            assert multiscale_mode in ['value', 'range']

        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio
        # TODO: refactor the override option in Resize
        self.override = override
        self.bbox_clip_border = bbox_clip_border 
    
    def random_select_scale(self):
        with tf.name_scope("random_select_scale"):
            img_scales = tf.convert_to_tensor(self.img_scale, tf.int32)
            scale = tf.random.shuffle(img_scales)[-1]

            return tf.cast(scale, tf.int32)
    
    def random_sample_scale(self):
        with tf.name_scope("random_sample_scale"):
            assert len(self.img_scale) == 2
            img_scales = tf.convert_to_tensor(self.img_scale, tf.float32)
            scale = tf.random.uniform([], tf.reduce_min(img_scales), tf.reduce_max(img_scales))

            return scale
    
    def random_sample_ratio(self):
        with tf.name_scope("random_select_ratio"):
            assert isinstance(self.img_scale, (tuple, list)) and len(self.img_scale) == 2
            min_ratio, max_ratio = self.ratio_range
            assert min_ratio <= max_ratio

            ratio = tf.random.uniform([], min_ratio, max_ratio)
            scale = tf.stack([self.img_scale[0] * ratio, self.img_scale[1] * ratio], 0)

            return tf.cast(scale, tf.int32)

    def _random_scale(self, image_info: dict):
        """Randomly sample an img_scale according to ``ratio_range`` and ``multiscale_mode``.

            If ``ratio_range`` is specified, a ratio will be sampled and be
            multiplied with ``img_scale``.
            If multiple scales are specified by ``img_scale``, a scale will be
            sampled according to ``multiscale_mode``.
            Otherwise, single scale will be used.

            Args:
                image_info (dict): Result dict .

            Returns:
                dict: Two new keys 'scale` are added into \
                    ``results``, which would be used by subsequent pipelines.
        """
        if self.ratio_range is not None:
            scale = self.random_sample_ratio()
        elif len(self.img_scale) == 1:
            scale = tf.convert_to_tensor(self.img_scale, tf.int32)
        elif self.multiscale_mode == "range":
            scale = self.random_sample_scale()
        elif self.multiscale_mode == "value":
            scale = self.random_select_scale()
        else:
            raise NotImplementedError

        image_info["scale"] = scale
    
    def _resize_img(self, image: tf.Tensor, image_info: dict) -> tf.Tensor:
        with tf.name_scope("resize_img"):
            if self.keep_ratio:
                img = resize_with_keep_ratio(image, image_info["scale"])
            else:
                img = tf.image.resize(image, image_info["scale"])
            
            old_hw = tf.cast(tf.shape(image)[:2], tf.float32)
            new_hw = tf.cast(tf.shape(img)[:2], tf.float32)
            hw_scale = new_hw / old_hw

            scale_factor = tf.convert_to_tensor([hw_scale[0], hw_scale[1]] * 2, tf.float32)
            image_info["scale_factor"] = scale_factor
            image_info["image_shape"] = new_hw
            image_info["padded_shape"] = new_hw
            image_info["keep_ratio"] = self.keep_ratio
        
        return img

    def _resize_boxes(self, image_info: dict):
        with tf.name_scope("resize_boxes"):
            boxes = image_info["boxes"] * image_info["scale_factor"]
            if self.bbox_clip_border:
                img_shape = image_info["image_shape"]
                boxes = tf.concat(
                    [
                        tf.clip_by_value(boxes[:, 0:1], 0., img_shape[1]),
                        tf.clip_by_value(boxes[:, 1:2], 0., img_shape[0]),
                        tf.clip_by_value(boxes[:, 2:3], 0., img_shape[1]),
                        tf.clip_by_value(boxes[:, 3:4], 0., img_shape[0])
                    ], axis=-1)
            
            image_info["boxes"] = boxes
    
    def _resize_mask(self, image_info: dict):
        if "mask" not in image_info:
            return
    
    def _resize_seg(self, image_info: dict):
        if "seg" not in image_info:
            return
    
    def __call__(self, image: tf.Tensor, image_info: dict) -> (tf.Tensor, dict):
        self._random_scale(image_info)
        image = self._resize_img(image, image_info)
        self._resize_boxes(image_info)
        self._resize_mask(image_info)
        self._resize_seg(image_info)

        return image, image_info


@AUGMENTATIONS.register
class FlipLeftToRight(object):
    def __init__(self, probability=0.5):
        self.prob = probability
   
    def _flip_with_mask(self, image, image_info):
        if should_apply_op(self.prob):
            image = tf.image.flip_left_right(image)
            orig_w = tf.shape(image)[1]
            orig_w = tf.cast(orig_w, tf.float32)

            boxes = image_info["boxes"]
            x1, y1, x2, y2 = tf.unstack(boxes, 4, 1)
            new_x1 = orig_w - x2
            new_x2 = orig_w - x1

            boxes = tf.stack([new_x1, y1, new_x2, y2], 1)
            image_info["boxes"] = boxes

            mask = image_info["mask"]
            mask = tf.map_fn(lambda inp: tf.image.flip_left_right(inp), mask, fn_output_signature=mask.dtype)
            image_info["mask"] = mask
                
        return image, image_info
    
    def _flip_without_mask(self, image, image_info):
        if should_apply_op(self.prob):
            image = tf.image.flip_left_right(image)
            orig_w = tf.shape(image)[1]
            orig_w = tf.cast(orig_w, tf.float32)

            boxes = image_info["boxes"]
            x1, y1, x2, y2 = tf.unstack(boxes, 4, 1)
            new_x1 = orig_w - x2
            new_x2 = orig_w - x1

            boxes = tf.stack([new_x1, y1, new_x2, y2], 1)
            image_info["boxes"] = boxes
            
        return image, image_info

    def __call__(self, image, image_info):
        with tf.name_scope("flip_left_to_right"):
            # if "mask" in image_info:
            #     return self._flip_with_mask(image, image_info)

            return self._flip_without_mask(image, image_info)


@AUGMENTATIONS.register
class SSDCrop(object):
    def __init__(self,
                 input_size,
                 patch_area_range=(0.3, 1.),
                 aspect_ratio_range=(0.5, 2.0),
                 min_overlaps=(0.1, 0.3, 0.5, 0.7, 0.9),
                 max_attempts=100,
                 **kwargs):
        self.input_size = input_size
        self.patch_area_range = patch_area_range
        self.aspect_ratio_range = aspect_ratio_range
        self.min_overlaps = tf.constant(min_overlaps, dtype=tf.float32)
        self.max_attempts = max_attempts

    def _random_overlaps(self):
        return tf.random.shuffle(self.min_overlaps)[0]

    def _random_crop(self, image, image_info):
        image_size = tf.shape(image)
        bbox_begin, bbox_size, distort_bbox = tf.image.sample_distorted_bounding_box(
            image_size=image_size,
            bounding_boxes=tf.expand_dims(image_info["boxes"], 0),
            min_object_covered=self._random_overlaps(),
            aspect_ratio_range=self.aspect_ratio_range,
            area_range=self.patch_area_range,
            max_attempts=self.max_attempts,
            use_image_if_no_bounding_boxes=True)

        patch = distort_bbox[0, 0]
        cropped_boxes, cropped_labels = clip_boxes_based_center(boxes, labels, patch)
        cropped_image = tf.slice(image, bbox_begin, bbox_size)
        cropped_image = tf.reshape(cropped_image, [bbox_size[0], bbox_size[1], 3])
        cropped_image = tf.image.resize(cropped_image, self.input_size)

        image_info["boxes"] = cropped_boxes
        image_info["labels"] = cropped_labels

        return cropped_image, image_info

    def __call__(self, image, image_info):
        with tf.name_scope("ssd_crop"):
            return self._random_crop(image, image_info)


class DataAnchorSampling(object):
    def __init__(self,
                 input_size=(640, 640),
                 anchor_scales=(16, 32, 64, 128, 256, 512),
                 overlap_threshold=0.7,
                 max_attempts=50,
                 **Kwargs):
        self.anchor_scales = tf.reshape(tf.constant(anchor_scales, tf.float32), [-1])
        self.num_scales = tf.shape(self.anchor_scales, out_type=tf.int64)[0]
        self.input_size = input_size
        self.overlap_threshold = tf.convert_to_tensor(overlap_threshold, tf.float32)

        self.mean = tf.convert_to_tensor([104, 117, 123], dtype=tf.float32)
        self.max_size = 12000
        self.inf = 9999999
        self.max_attempts = max_attempts

    def _sampling(self, image, image_info):
        image_shape = tf.shape(image)
        # image_size = tf.convert_to_tensor([image_shape[0],
        #                                    image_shape[1],
        #                                    image_shape[0],
        #                                    image_shape[1]], tf.float32)
        # boxes = boxes * image_size
        image_size = tf.cast(image_shape[0:2], tf.float32)
        height, width = image_size[0], image_size[1]
        boxes = image_info["boxes"]
        box_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) * height * width
        box_areas = box_areas[box_areas > 0]

        if tf.less_equal(tf.size(box_areas), 0):
            image = tf.image.resize(image, self.input_size)
            return image, image_info

        rand_box_idx = tf.random.uniform([], 0, tf.shape(box_areas)[0], dtype=tf.int32)
        rand_box_scale = tf.math.sqrt(box_areas[rand_box_idx])

        anchor_idx = tf.argmin(tf.math.abs(self.anchor_scales - rand_box_scale))
        anchor_idx_range = tf.minimum(anchor_idx + 1, self.num_scales) + 1
        target_anchor = tf.random.shuffle(self.anchor_scales[0:anchor_idx_range])[-1]
        ratio = target_anchor / rand_box_scale
        ratio *= (tf.math.pow(2., tf.random.uniform([], -1, 1, dtype=tf.float32)))

        if tf.greater(height * ratio * width * ratio, self.max_size * self.max_size):
            ratio = (self.max_size * self.max_size / (height * width)) ** 0.5
        else:
            ratio = ratio

        resizing_height = height * ratio
        resizing_width = width * ratio
        resizing_size = tf.convert_to_tensor([resizing_height, resizing_width], tf.int32)
        image = tf.image.resize(image, resizing_size)

        y1 = boxes[rand_box_idx, 0] * height
        x1 = boxes[rand_box_idx, 1] * width
        y2 = boxes[rand_box_idx, 2] * height
        x2 = boxes[rand_box_idx, 3] * width
        rand_box_w = x2 - x1 + 1
        rand_box_h = y2 - y1 + 1
        inp_h = tf.cast(self.input_size[0], tf.float32)
        inp_w = tf.cast(self.input_size[1], tf.float32)

        sample_boxes = tf.TensorArray(size=50, dtype=boxes.dtype)
        for i in tf.range(50):
            if tf.less(inp_w, tf.maximum(resizing_height, resizing_width)):
                if tf.less_equal(rand_box_w, inp_w):
                    offset_width = tf.random.uniform([], x1 + rand_box_w - inp_w, x1)
                else:
                    offset_width = tf.random.uniform([], x1, x1 + rand_box_w - inp_w)
                if tf.less_equal(rand_box_h, inp_h):
                    offset_height = tf.random.uniform([], y1 + rand_box_h - inp_h, y1)
                else:
                    offset_height = tf.random.uniform([], y1, y1 + rand_box_h - inp_h)
            else:
                offset_height = tf.random.uniform([], resizing_height - inp_h, 0)
                offset_width = tf.random.uniform([], resizing_width - inp_w, 0)
            offset_height = tf.math.floor(offset_height)
            offset_width = tf.math.floor(offset_width)

            patch = tf.convert_to_tensor([offset_height / height, 
                                          offset_width / width, 
                                          (offset_height + inp_h) / height, 
                                          (offset_width + inp_w) / width], tf.float32)
            in_patch = is_box_center_in_patch(boxes, patch)
            overlaps = jaccard(boxes, patch)

            if tf.logical_or(tf.reduce_any(in_patch), tf.greater_equal(tf.reduce_max(overlaps), 0.7)):
                sample_boxes = sample_boxes.write(i, patch)
            else:
                continue
        
        sample_boxes = sample_boxes.stack()
        if tf.greater(tf.size(sample_boxes), 0):
            choice_patch = tf.random.shuffle(sample_boxes)[0]
            current_boxes, current_labels = clip_boxes_based_center(boxes, labels, choice_patch)

            if tf.logical_or(tf.less(choice_patch[0], 0), tf.less(choice_patch[1], 0)):
                if tf.greater_equal(choice_patch[0], 0):
                    top_padding = tf.zeros([], tf.int32)
                    offset_height = tf.cast(choice_patch[0], tf.int32)
                else:
                    # new_img_width = resizing_width - choice_patch[0]
                    top_padding = tf.cast(-1. * choice_patch[0], tf.int32)
                    offset_height = tf.zeros([], tf.int32)
                if tf.greater_equal(choice_patch[1], 0):
                    left_padding = tf.zeros([], tf.int32)
                    offset_width = tf.cast(choice_patch[1], tf.int32)
                else:
                    # new_img_height = resizing_height - choice_patch[1]
                    left_padding = tf.cast(-1. * choice_patch[1], tf.int32)
                    offset_width = tf.zeros([], tf.int32)

                # bottom_padding = tf.maximum(tf.cast(inp_h, tf.int32) - top_padding - resizing_size[0], 0)
                # right_padding = tf.maximum(tf.cast(inp_w, tf.int32) - left_padding - resizing_size[1], 0)
            else:
                left_padding = tf.zeros([], tf.int32)
                top_padding = tf.zeros([], tf.int32)
                offset_height = tf.cast(choice_patch[0], tf.int32)
                offset_width = tf.cast(choice_patch[1], tf.int32)

            bottom_padding = tf.minimum(resizing_size[0] - tf.cast(inp_h, tf.int32) - offset_height, 0) * -1
            right_padding = tf.minimum(resizing_size[1] - tf.cast(inp_w, tf.int32) - offset_width, 0) * -1
            target_height = tf.cast(choice_patch[2] - choice_patch[0], tf.int32)
            target_width = tf.cast(choice_patch[3] - choice_patch[1], tf.int32)

            # if tf.logical_or(choice_patch[0] + choice_patch[2] <= 2, choice_patch[1] + choice_patch[3] <= 2):
                # tf.print(choice_patch, [offset_height, offset_width, target_height, target_width],
                        #  [top_padding, bottom_padding, left_padding, right_padding], resizing_size, tf.shape(image))
            
            if tf.logical_and(target_width > 0, target_height > 0):
                padded_image = tf.pad(image,
                                  paddings=[[top_padding, bottom_padding], [left_padding, right_padding], [0, 0]],
                                  constant_values=128)
                cropped_image = tf.image.crop_to_bounding_box(image=padded_image,
                                                              offset_height=offset_height,
                                                              offset_width=offset_width,
                                                              target_height=target_height,
                                                              target_width=target_width)

                return cropped_image, current_boxes, current_labels
            else:
                image = tf.image.resize(image, self.input_size)
                return image, boxes, labels
        else:
            image = tf.image.resize(image, self.input_size)
            return image, boxes, labels

    def __call__(self, image, image_info):
        with tf.name_scope("data_anchor_sampling"):
            image, boxes, labels = self._sampling(image, image_info)

            image_info["boxes"] = boxes
            image_info["labels"] = labels

            return image_info


@AUGMENTATIONS.register
class RandomDistortColor(object):
    def __init__(self,
                 brightness=32./255.,
                 min_saturation=0.5,
                 max_saturation=1.5,
                 hue=0.2,
                 min_contrast=0.5,
                 max_contrast=1.5,
                 **Kwargs):
        self.brightness = brightness
        self.min_saturation = min_saturation
        self.max_saturation = max_saturation
        self.hue = hue
        self.min_contrast = min_contrast
        self.max_contrast = max_contrast

    def _distort_color0(self, image):
        image = tf.image.random_brightness(image, max_delta=self.brightness)
        image = tf.image.random_saturation(image, lower=self.min_saturation, upper=self.max_saturation)
        image = tf.image.random_hue(image, max_delta=self.hue)
        image = tf.image.random_contrast(image, lower=self.min_contrast, upper=self.max_contrast)

        return image

    def _distort_color1(self, image):
        image = tf.image.random_saturation(image, lower=self.min_saturation, upper=self.max_saturation)
        image = tf.image.random_brightness(image, max_delta=self.brightness)
        image = tf.image.random_contrast(image, lower=self.min_contrast, upper=self.max_contrast)
        image = tf.image.random_hue(image, max_delta=self.hue)

        return image
    
    def _distort_color2(self, image):
        image = tf.image.random_contrast(image, lower=self.min_contrast, upper=self.max_contrast)
        image = tf.image.random_hue(image, max_delta=self.hue)
        image = tf.image.random_brightness(image, max_delta=self.brightness)
        image = tf.image.random_saturation(image, lower=self.min_saturation, upper=self.max_saturation)

        return image
    
    def _distort_color3(self, image):
        image = tf.image.random_hue(image, max_delta=self.hue)
        image = tf.image.random_saturation(image, lower=self.min_saturation, upper=self.max_saturation)
        image = tf.image.random_contrast(image, lower=self.min_contrast, upper=self.max_contrast)
        image = tf.image.random_brightness(image, max_delta=self.brightness)

        return image

    def __call__(self, image, image_info):
        with tf.name_scope("distort_color"):
            rand_int = tf.random.uniform([], 0, 4, tf.int32)
            if rand_int == 0:
                image = self._distort_color0(image)
            elif rand_int == 1:
                image = self._distort_color1(image)
            elif rand_int == 2:
                image = self._distort_color2(image)
            else:
                image = self._distort_color3(image)
            
            image = tf.minimum(tf.maximum(image, 0), 255)

            return image, image_info


@AUGMENTATIONS.register
class RandCropOrPad(object):
    def __init__(self, size, clip_box_base_center=True):
        self._crop_size = size
        self.clip_box_base_center = clip_box_base_center
   
    def _rand_resize_and_crop(self, image, image_info):
        with tf.name_scope("resize_and_crop"):
            scaled_shape = tf.cast(tf.shape(image), tf.int32)

            off_y = tf.random.uniform([], 0, tf.maximum(scaled_shape[0] - self._crop_size[0], 1), dtype=tf.int32)
            off_x = tf.random.uniform([], 0, tf.maximum(scaled_shape[1] - self._crop_size[1], 1), dtype=tf.int32)
            tgt_h = tf.minimum(self._crop_size[0], scaled_shape[0])
            tgt_w = tf.minimum(self._crop_size[1], scaled_shape[1])
            image = tf.image.crop_to_bounding_box(image, off_y, off_x, tgt_h, tgt_w)
            
            image = tf.cond(
                tf.logical_or(tgt_h < self._crop_size[0], tgt_w < self._crop_size[1]),
                lambda: tf.pad(image, [[0, self._crop_size[0] - tgt_h], [0, self._crop_size[1] - tgt_w], [0, 0]]),
                lambda: image)

            boxes = image_info["boxes"]
            labels = image_info["labels"]
            boxes, labels = clip_boxes(boxes, labels, [off_x, off_y, off_x + tgt_w, off_y + tgt_h], self.clip_box_base_center)

            image_info["boxes"] = boxes
            image_info["labels"] = labels
            image_info["image_shape"] = tf.convert_to_tensor(self._crop_size, tf.float32)
            
            return image

    def __call__(self, image, image_info):
        image = self._rand_resize_and_crop(image, image_info)

        return image, image_info


@AUGMENTATIONS.register
class Pad(object):
    def __init__(self, size=None, size_divisor=None, pad_value=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_value = pad_value
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None
    
    def _pad_img(self, image: tf.Tensor, image_info: dict) -> tf.Tensor:
        with tf.name_scope("pad_img"):
            img_shape = tf.shape(image)
            size = self.size
            if size is not None:
                paddings = tf.convert_to_tensor(
                    [[0, size[0] - img_shape[0]], [0, size[1] - img_shape[1]], [0, 0]], 
                    tf.int32)
                
            else:
                float_img_shape = tf.cast(img_shape, tf.float32)
                size_h = tf.cast(tf.math.ceil(float_img_shape[0] / self.size_divisor) * self.size_divisor, tf.int32)
                size_w = tf.cast(tf.math.ceil(float_img_shape[1] / self.size_divisor) * self.size_divisor, tf.int32)
                size = tf.convert_to_tensor([size_h, size_w], tf.float32)
                paddings = tf.convert_to_tensor(
                    [[0, size_h - img_shape[0]], [0, size_w - img_shape[1]], [0, 0]], 
                    tf.int32)
            padded = tf.pad(image, paddings)

            image_info["image_shape"] = tf.cast(size, tf.float32)

            return padded
    
    def _pad_mask(self, image_info):
        pass

    def _pad_seg(self, image_info):
        pass

    def __call__(self, image, image_info):
        image = self._pad_img(image, image_info)
        self._pad_mask(image_info)
        self._pad_seg(image_info)

        return image, image_info
