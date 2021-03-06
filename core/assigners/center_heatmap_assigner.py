import math
import numpy as np
import tensorflow as tf
from .assigner import Assigner
from ..builder import ASSIGNERS


@ASSIGNERS.register
class CenterHeatmapAssigner(Assigner):
    def __init__(self, strides=4, num_classes=80, max_objs=128, **kwargs):
        super(CenterHeatmapAssigner, self).__init__(**kwargs)
        
        self.strides = strides
        self.num_classes = num_classes
        self.max_objs = max_objs

    def gaussian_radius(self, height, width, min_overlap=0.7):
        r"""Generate 2D gaussian radius.

        This function is modified from the `official github repo
        <https://github.com/princeton-vl/CornerNet-Lite/blob/master/core/sample/
        utils.py#L65>`_.

        Given ``min_overlap``, radius could computed by a quadratic equation
        according to Vieta's formulas.

        There are 3 cases for computing gaussian radius, details are following:

        - Explanation of figure: ``lt`` and ``br`` indicates the left-top and
        bottom-right corner of ground truth box. ``x`` indicates the
        generated corner at the limited position when ``radius=r``.

        - Case1: one corner is inside the gt box and the other is outside.

        .. code:: text

            |<   width   >|

            lt-+----------+         -
            |  |          |         ^
            +--x----------+--+
            |  |          |  |
            |  |          |  |    height
            |  | overlap  |  |
            |  |          |  |
            |  |          |  |      v
            +--+---------br--+      -
            |          |  |
            +----------+--x

        To ensure IoU of generated box and gt box is larger than ``min_overlap``:

        .. math::
            \cfrac{(w-r)*(h-r)}{w*h+(w+h)r-r^2} \ge {iou} \quad\Rightarrow\quad
            {r^2-(w+h)r+\cfrac{1-iou}{1+iou}*w*h} \ge 0 \\
            {a} = 1,\quad{b} = {-(w+h)},\quad{c} = {\cfrac{1-iou}{1+iou}*w*h}
            {r} \le \cfrac{-b-\sqrt{b^2-4*a*c}}{2*a}

        - Case2: both two corners are inside the gt box.

        .. code:: text

            |<   width   >|

            lt-+----------+         -
            |  |          |         ^
            +--x-------+  |
            |  |       |  |
            |  |overlap|  |       height
            |  |       |  |
            |  +-------x--+
            |          |  |         v
            +----------+-br         -

        To ensure IoU of generated box and gt box is larger than ``min_overlap``:

        .. math::
            \cfrac{(w-2*r)*(h-2*r)}{w*h} \ge {iou} \quad\Rightarrow\quad
            {4r^2-2(w+h)r+(1-iou)*w*h} \ge 0 \\
            {a} = 4,\quad {b} = {-2(w+h)},\quad {c} = {(1-iou)*w*h}
            {r} \le \cfrac{-b-\sqrt{b^2-4*a*c}}{2*a}

        - Case3: both two corners are outside the gt box.

        .. code:: text

            |<   width   >|

            x--+----------------+
            |  |                |
            +-lt-------------+  |   -
            |  |             |  |   ^
            |  |             |  |
            |  |   overlap   |  | height
            |  |             |  |
            |  |             |  |   v
            |  +------------br--+   -
            |                |  |
            +----------------+--x

        To ensure IoU of generated box and gt box is larger than ``min_overlap``:

        .. math::
            \cfrac{w*h}{(w+2*r)*(h+2*r)} \ge {iou} \quad\Rightarrow\quad
            {4*iou*r^2+2*iou*(w+h)r+(iou-1)*w*h} \le 0 \\
            {a} = {4*iou},\quad {b} = {2*iou*(w+h)},\quad {c} = {(iou-1)*w*h} \\
            {r} \le \cfrac{-b+\sqrt{b^2-4*a*c}}{2*a}

        Args:
            det_size (list[int]): Shape of object.
            min_overlap (float): Min IoU with ground truth for boxes generated by
                keypoints inside the gaussian kernel.

        Returns:
            radius (int): Radius of gaussian kernel.
        """
        a1  = 1
        b1  = (height + width)
        c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
        sq1 = tf.math.sqrt(b1 ** 2 - 4 * a1 * c1)
        r1  = (b1 + sq1) / 2

        a2  = 4
        b2  = 2 * (height + width)
        c2  = (1 - min_overlap) * width * height
        sq2 = tf.math.sqrt(b2 ** 2 - 4 * a2 * c2)
        r2  = (b2 + sq2) / 2

        a3  = 4 * min_overlap
        b3  = -2 * min_overlap * (height + width)
        c3  = (min_overlap - 1) * width * height
        sq3 = tf.math.sqrt(b3 ** 2 - 4 * a3 * c3)
        r3  = (b3 + sq3) / 2

        return tf.minimum(tf.minimum(r1, r2), r3)

    def gaussian_2d(self, height, width, sigma=1):
        m = (height - 1.) / 2. 
        n = (width - 1.) / 2.
        
        y = tf.expand_dims(tf.range(-m, m+1), -1)
        x = tf.expand_dims(tf.range(-n, n+1), 0)

        h = tf.math.exp(-(x * x + y * y) / (2 * sigma * sigma))
        h = tf.where(h < tf.reduce_max(h) * np.finfo(np.float32).eps, tf.zeros_like(h), h)

        return h

    def draw_umich_gaussian(self, heatmap, class_index, center, radius, k=1):
        diameter = 2 * radius + 1
        gaussian = self.gaussian_2d(diameter, diameter, sigma=diameter / 6)
        
        y, x = center[0], center[1]
        height, width = tf.shape(heatmap)[0], tf.shape(heatmap)[1]
        y = tf.cast(y, tf.int32)
        x = tf.cast(x, tf.int32)
        radius = tf.cast(radius, tf.int32)
        height = tf.cast(height, tf.int32)
        width = tf.cast(width, tf.int32)
        class_index = tf.cast(class_index, tf.int32)
            
        left, right = tf.minimum(x, radius), tf.minimum(width - x, radius + 1)
        top, bottom = tf.minimum(y, radius), tf.minimum(height - y, radius + 1)

        # masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
        gaussian = tf.expand_dims(gaussian, -1)
        masked_gaussian = gaussian[radius - top: radius + bottom, radius - left: radius + right, :]

        paddings = tf.convert_to_tensor([[y - top, height - y - bottom],
                                         [x - left, width - x - right],
                                         [class_index, self.num_classes - class_index - 1]])
        padded_gaussian = tf.pad(masked_gaussian, paddings)
        heatmap = tf.maximum(heatmap, padded_gaussian * k)
        # if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: # TODO debug
        #     np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
        return heatmap

    def draw_dense_reg(self, regmap, heatmap, center, value, radius, is_offset=False):
        diameter = 2 * radius + 1
        gaussian = self.gaussian_2d(diameter, diameter, sigma=diameter / 6)

        value = tf.reshape(tf.convert_to_tensor([value], dtype=tf.float32), (1, 1, -1))
        dim = tf.shape(value)[-1]
        reg = tf.ones((diameter * 2 + 1, diameter * 2 + 1, dim), dtype=tf.float32) * value
        
        # if tf.logical_and(is_offset, tf.equal(dim, 2)):
        #     delta = tf.range(diameter * 2 + 1) - radius
        #     # reg[0] = reg[0] - tf.reshape(delta, (1, -1))
        #     # reg[1] = reg[1] - tf.reshape(delta, (-1, 1))
        
        y, x = center[0], center[1]
        height, width = tf.shape(heatmap)[0], tf.shape(heatmap)[1]
        y = tf.cast(y, tf.int32)
        x = tf.cast(x, tf.int32)
        radius = tf.cast(radius, tf.int32)
        height = tf.cast(height, tf.int32)
        width = tf.cast(width, tf.int32)
            
        left, right = tf.minimum(x, radius), tf.minimum(width - x, radius + 1)
        top, bottom = tf.minimum(y, radius), tf.minimum(height - y, radius + 1)

        # masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
        # masked_regmap = regmap[y - top:y + bottom, x - left:x + right, :]
        masked_gaussian = gaussian[radius - top: radius + bottom, radius - left: radius + right]
        masked_reg = reg[radius - top: radius + bottom, radius - left: radius + right, :]

        reg_paddings = tf.convert_to_tensor([[y - top, height - y - bottom], [x - left, width - x - right], [0, 0]])
        padded_reg = tf.pad(masked_reg, reg_paddings)
        padded_gaussian = tf.pad(masked_gaussian, reg_paddings[:2])
        idx = tf.expand_dims(tf.cast(padded_gaussian >= heatmap, regmap.dtype), -1)
        regmap = (1 - idx) * regmap + idx * padded_reg
        # if tf.logical_and(tf.reduce_min(tf.shape(masked_gaussian)) > 0, tf.reduce_min(tf.shape(masked_heatmap)) > 0): # TODO debug
        #     idx = tf.reshape((masked_gaussian >= masked_heatmap),
        #         [tf.shape(masked_gaussian)[0], tf.shape(masked_gaussian)[1], 1])
        #     idx = tf.cast(idx, masked_regmap)
        #     masked_regmap = (1 - idx) * masked_regmap + idx * masked_reg

        # regmap[y - top:y + bottom, x - left:x + right, :] = masked_regmap

        return regmap

    def draw_msra_gaussian(self, heatmap, center, sigma):
        tmp_size = sigma * 3
        mu_x = int(center[0] + 0.5)
        mu_y = int(center[1] + 0.5)
        w, h = heatmap.shape[0], heatmap.shape[1]
        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
        br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]

        if ul[0] >= h or ul[1] >= w or br[0] < 0 or br[1] < 0:
            return heatmap
        
        size = 2 * tmp_size + 1
        x = np.arange(0, size, 1, np.float32)
        y = x[:, np.newaxis]
        x0 = y0 = size // 2
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
        g_x = max(0, -ul[0]), min(br[0], h) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], w) - ul[1]
        img_x = max(0, ul[0]), min(br[0], h)
        img_y = max(0, ul[1]), min(br[1], w)

        heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
            heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]],
            g[g_y[0]:g_y[1], g_x[0]:g_x[1]])

        return heatmap
    
    @tf.function
    def assign(self, heatmap, wh_targets, center_offset, boxes, labels):
        """Assign heatmap using numpy.
        
        Args:
            heatmap (np.ndarray): the init heatmap, shape is (output_height, output_width, num_classes).
            wh_targets (np.ndarray): the regression target of height and width, shape is (output_height, output_width, 2).
            center_offset (np.ndarray): the offset of center, shape is (max_objs, 2).
            boxes (np.ndarray): the ground-truth boxes, e.g. [y1, x1, y2, x2].
            labels (np.ndarray): the labels, start from 0.
        
        Returns:
            heatmap, wh_targets, wh_mask, center_offset
        """
        num_objs = tf.minimum(tf.shape(boxes)[0], self.max_objs)
        for i in tf.range(num_objs):
            box = boxes[i] / self.strides
            cls_ind = labels[i]
            w, h = box[2] - box[0], box[3] - box[1]
            if tf.logical_and(h > 0, w > 0):
                radius = self.gaussian_radius(tf.math.ceil(h), tf.math.ceil(w))
                radius = tf.maximum(0., tf.math.floor(radius))

                center = (box[0:2] + box[2:4]) * 0.5
                center = center[::-1]
                center_float = tf.math.floor(center)
                heatmap = self.draw_umich_gaussian(heatmap, cls_ind, center_float, radius)
                wh = tf.convert_to_tensor([1. * w, 1. * h], center_offset.dtype)
                
                offset = center - center_float
                center_int = tf.expand_dims(tf.expand_dims(tf.cast(center_float, tf.int32), 0), 0)
                center_offset = tf.tensor_scatter_nd_update(center_offset, center_int, [[offset]])
                
                wh_targets = self.draw_dense_reg(wh_targets, tf.reduce_max(heatmap, axis=2), center_float, wh, radius)
        
        # wh_mask = np.concatenate([heatmap.max(axis=2, keepdims=True)] * 2, axis=2)
        
        return heatmap, wh_targets, center_offset
    
    # @tf.function
    # def assign(self, heatmap, wh_targets, center_offset, boxes, labels):
    #     return tf.numpy_function(
    #         self.assign_np,
    #         inp=(heatmap, wh_targets, center_offset, boxes, labels),
    #         Tout=(tf.float32, tf.float32, tf.float32))
       
    
    def __call__(self, feat_height, feat_width, boxes, labels):
        heatmap = tf.zeros([feat_height, feat_width, self.num_classes], tf.float32)
        wh_targets = tf.zeros([feat_height, feat_width, 2], tf.float32)
        center_offset = tf.zeros([feat_height, feat_width, 2], tf.float32)
        # center_indices = tf.zeros([self.max_objs], tf.int32)
        # center_mask = tf.zeros([self.max_objs], tf.int32)

        valid = labels >= 0
        labels = tf.boolean_mask(labels, valid)
        boxes = tf.boolean_mask(boxes, valid)
        # heatmap, wh_targets, center_offset, center_indices, center_mask = self.assign(
        #     heatmap, wh_targets, center_offset, center_indices, center_mask, boxes, labels)
        heatmap, wh_targets, center_offset = self.assign(
            heatmap, wh_targets, center_offset, boxes, labels)

        heatmap = tf.reshape(heatmap, [feat_height, feat_width, self.num_classes])
        wh_targets = tf.reshape(wh_targets, [feat_height, feat_width, 2])
        center_offset = tf.reshape(center_offset, [feat_height, feat_width, 2])
        # center_indices = tf.reshape(center_indices, [self.max_objs])
        # center_mask = tf.reshape(center_mask, [self.max_objs])

        return heatmap, wh_targets, center_offset



def test():
    import cv2
    boxes = tf.constant([[32, 120, 120, 256], [200, 201, 434, 472]], tf.float32)
    labels = tf.constant([0, 23], tf.int32)
    
    assigner = CenterHeatmapAssigner(4, 80, 128)
    heatmap, wh_targets, center_offset = assigner(boxes, labels)
    print(heatmap.shape)

    heat23 = (heatmap[..., 23:24] * 255).numpy().astype(np.uint8)
    cv2.imshow("heat23", heat23)
    cv2.waitKey(0)

    print(center_offset.shape)
    indices = tf.where(tf.reduce_max(heatmap, -1) == 1)
    print(indices)
    heatmap2 = tf.reduce_max(heatmap * 255, -1, keepdims=True)
    center_offset = tf.gather_nd(center_offset, indices)
    show_heatmap = heatmap2.numpy().astype(np.uint8)
    print(center_offset)
    yx = tf.where(tf.reduce_max(heatmap, -1) == 1).numpy()
    wh = tf.boolean_mask(wh_targets, tf.reduce_max(heatmap, -1) == 1).numpy()  
        
    boxes = boxes.numpy() / 4
    for i in range(len(yx)):
        print(yx[i], wh[i])
        show_heatmap = cv2.rectangle(
            show_heatmap, 
            (int(yx[i, 1] - wh[i, 0] / 2), int(yx[i, 0] - wh[i, 1] / 2)),
            (int(yx[i, 1] + wh[i, 0] / 2), int(yx[i, 0] + wh[i, 1] / 2)),
            (255, 255, 255))
        
        # box = boxes[i]
        # show_heatmap = cv2.rectangle(show_heatmap, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0))

    cv2.imshow("heatmap", show_heatmap)
    cv2.waitKey(0)


if __name__ == "__main__":
    test()