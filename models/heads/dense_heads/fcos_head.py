import math
import tensorflow as tf
from core import build_loss
from ...builder import HEADS
from ..head import BaseHead 
from core.layers import Scale
from core.layers import build_activation
from core.layers import build_normalization


@HEADS.register
class FCOSHead(BaseHead):
    def __init__(self, cfg, num_classes, **kwargs):
        super(FCOSHead, self).__init__(cfg, num_classes=num_classes, **kwargs)
    
        self._make_shared_convs()
        self.classifier = tf.keras.layers.Conv2D(
            filters=cfg.num_classes,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            use_bias=True,
            bias_initializer=tf.keras.initializers.Constant(
            -math.log((1. - cfg.prior) / cfg.prior)),
            name="predicted_class")
        self.regressor = tf.keras.layers.Conv2D(
            filters=4,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            use_bias=True,
            name="predicted_box")
        self.centerness = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=(3, 3),
            strides=(1, 1),
            use_bias=True,
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
            padding="same",
            name="predicted_centerness")
        self.scales = [
            Scale(value=1.0, name="%d" % i) 
            for i, _ in enumerate(range(cfg.min_level, cfg.max_level + 1))
        ]

        self.normalize_box = cfg.normalize_box
        
        self.centerness_loss_func = build_loss(**cfg.centerness_loss.as_dict())
    
    def _make_shared_convs(self):
        self.box_shared_convs = tf.keras.Sequential(name="box_net")
        self.class_shared_convs = tf.keras.Sequential(name="cls_net")

        i = 0
        for _ in range(self.cfg.repeats):
            self.box_shared_convs.add(
                tf.keras.layers.Conv2D(filters=self.cfg.feat_dims,
                                       kernel_size=(3, 3),
                                       padding="same",
                                       strides=(1, 1),
                                       use_bias=True,
                                       name="%d" % i))
            self.class_shared_convs.add(
                tf.keras.layers.Conv2D(filters=self.cfg.feat_dims,
                                       kernel_size=(3, 3),
                                       strides=(1, 1),
                                       padding="same",
                                       use_bias=True,
                                       name="%d" % i))
            i += 1
            self.box_shared_convs.add(
                build_normalization(name="%d" % i, **self.cfg.normalization.as_dict()))
            self.class_shared_convs.add(
                build_normalization(name="%d" % i, **self.cfg.normalization.as_dict()))
            i += 1
            self.box_shared_convs.add(
                build_activation(name="%d" % i, **self.cfg.activation.as_dict()))
            self.class_shared_convs.add(
                build_activation(name="%d" % i, **self.cfg.activation.as_dict()))
            i += 1

    def call(self, inputs, training=None):
        predicted_boxes = dict()
        predicted_labels = dict()
        predicted_centerness = dict()

        for i, level in enumerate(range(self.min_level, self.max_level + 1)):
            box_feat = self.box_shared_convs(inputs[i], training=training)
            label_feat = self.class_shared_convs(inputs[i], training=training)
    
            pred_boxes = self.regressor(box_feat)
            pred_boxes = self.scales[i](pred_boxes)
            
            pred_labels = self.classifier(label_feat)

            if self.cfg.centerness_on_box:
                pred_centerness = self.centerness(box_feat)
            else:
                pred_centerness = self.centerness(label_feat)

            if self.cfg.normalize_box:
                pred_boxes = tf.nn.relu(pred_boxes) * (2 ** level)
            else:
                pred_boxes = tf.math.exp(pred_boxes)

            predicted_boxes["level%d" % level] = pred_boxes
            predicted_labels["level%d" % level] = pred_labels
            predicted_centerness["level%d" % level] = pred_centerness

        outputs = dict(boxes=predicted_boxes, 
                       labels=predicted_labels, 
                        centerness=predicted_centerness)
        
        if self.is_training:
            return outputs
        
        return self.get_boxes(outputs)
    
    def get_targets(self, gt_boxes, gt_labels, grid_y, grid_x, strides, object_size_of_interest):
        with tf.name_scope("gt_targets"):
            batch_size = tf.shape(gt_boxes)[0]
            target_boxes_ta = tf.TensorArray(tf.float32, batch_size)
            target_labels_ta = tf.TensorArray(tf.int64, batch_size)
            target_centerness_ta = tf.TensorArray(tf.float32, batch_size)
            label_weights_ta = tf.TensorArray(tf.float32, batch_size)
            box_weights_ta = tf.TensorArray(tf.float32, batch_size)
            
            grid_y = tf.cast(grid_y, tf.float32)
            grid_x = tf.cast(grid_x, tf.float32) 

            for i in tf.range(batch_size):
                t_boxes, t_labels, t_centerness = self.assigner(
                    gt_boxes[i], gt_labels[i], grid_y, grid_x, strides, object_size_of_interest)

                t_boxes, t_labels, b_weights, l_weights = self.sampler(t_boxes, t_labels)

                target_boxes_ta = target_boxes_ta.write(i, t_boxes)
                target_labels_ta = target_labels_ta.write(i, t_labels)
                target_centerness_ta = target_centerness_ta.write(i, t_centerness)
                box_weights_ta = box_weights_ta.write(i, b_weights)
                label_weights_ta = label_weights_ta.write(i, l_weights)
            
            target_boxes = target_boxes_ta.stack()
            target_labels = target_labels_ta.stack()
            target_centerness = target_centerness_ta.stack()
            box_weights = box_weights_ta.stack()
            label_weights = label_weights_ta.stack()

            target_boxes = tf.stop_gradient(target_boxes)
            target_labels = tf.stop_gradient(target_labels)
            box_weights = tf.stop_gradient(box_weights)
            label_weights = tf.stop_gradient(label_weights)
            target_centerness = tf.stop_gradient(target_centerness)

            return target_boxes, target_labels, target_centerness, box_weights, label_weights
    
    def get_grid_points(self, feat_height, feat_width, stride):
        with tf.name_scope("get_grid_points"):
            grid_x, grid_y = tf.meshgrid(tf.range(0, feat_width), tf.range(0, feat_width))
            grid_y = (tf.cast(grid_y, tf.float32)  + 0.5) * stride
            grid_x = (tf.cast(grid_x, tf.float32)  + 0.5) * stride

            return grid_y, grid_x

    def compute_losses(self, predictions, image_info):
        with tf.name_scope("compute_losses"):
            avg_factor_list = []
            bbox_loss_avg_factor_list = []
            bbox_loss_list = []
            label_loss_list = []
            centerness_loss_list = []

            predicted_boxes = predictions["boxes"]
            predicted_labels = predictions["labels"]
            predicted_centerness = predictions["centerness"]
            
            for i, level in enumerate(range(self.min_level, self.max_level+1)):
                pred_boxes = tf.cast(predicted_boxes["level%d" % level], tf.float32)
                pred_labels = tf.cast(predicted_labels["level%d" % level], tf.float32)
                pred_centerness = tf.cast(predicted_centerness["level%d" % level], tf.float32)

                h, w = tf.shape(pred_boxes)[1], tf.shape(pred_boxes)[2]
                grid_ys, grid_xs = self.get_grid_points(h, w, 2 ** level)
                # pred_boxes = self.bbox_decoder(pred_boxes, grid_ys, grid_xs)
                grid_ys = tf.cast(grid_ys, pred_boxes.dtype)
                grid_xs = tf.cast(grid_xs, pred_boxes.dtype)
                pred_boxes = tf.stack([grid_ys - pred_boxes[..., 1],
                                       grid_xs - pred_boxes[..., 0],
                                       grid_ys + pred_boxes[..., 3],
                                       grid_xs + pred_boxes[..., 2]], axis=-1)

                pred_boxes = tf.reshape(pred_boxes, [-1, h * w, 4])
                pred_labels = tf.reshape(pred_labels, [-1, h * w, self._label_dims])
                pred_centerness = tf.reshape(pred_centerness, [-1, h * w, 1])
                
                gt_boxes = tf.cast(image_info["boxes"], tf.float32)
                gt_labels = tf.cast(image_info["labels"], tf.int64)

                target_boxes, target_labels, target_centerness, box_weights, label_weights = self.get_targets(
                   gt_boxes, gt_labels, grid_ys, grid_xs, 2 ** level, self.cfg.head.object_sizes_of_interest[i])
                        
                if self.use_sigmoid:
                    one_hot_target_labels = tf.one_hot(target_labels - 1, self.num_classes)
                else:
                    one_hot_target_labels = tf.one_hot(target_labels, self.num_classes + 1)

                centerness_weights = tf.squeeze(target_centerness, -1) * box_weights
                label_loss = self.label_loss_func(one_hot_target_labels, pred_labels, label_weights) 
                bbox_loss = self.bbox_loss_func(target_boxes, pred_boxes, centerness_weights)
                centerness_loss = self.centerness_loss_func(target_centerness, pred_centerness, box_weights)

                factor = tf.reduce_sum(box_weights)
                bbox_factor = tf.reduce_sum(centerness_weights)

                avg_factor_list.append(factor)
                bbox_loss_avg_factor_list.append(bbox_factor)
                bbox_loss_list.append(bbox_loss)
                label_loss_list.append(label_loss)
                centerness_loss_list.append(centerness_loss)
            
            avg_factor = tf.add_n(avg_factor_list) + 1.
            bbox_loss_avg_factor = tf.add_n(bbox_loss_avg_factor_list) + 1.
            bbox_loss = tf.add_n(bbox_loss_list) / bbox_loss_avg_factor
            label_loss = tf.add_n(label_loss_list) / avg_factor
            centerness_loss = tf.add_n(centerness_loss_list) / avg_factor

            return dict(bbox_loss=bbox_loss, label_loss=label_loss, centerness_loss=centerness_loss)
     
    def get_boxes(self, outputs):
        with  tf.name_scope("get_boxes"):
            predicted_boxes_list = []
            predicted_labels_list = []
            predicted_centerness_list = []
            for _, level in enumerate(range(self.min_level, self.max_level + 1)):
                pred_boxes = tf.cast(outputs["boxes"]["level%d" % level], tf.float32)
                pred_labels = tf.cast(outputs["labels"]["level%d" % level], tf.float32)
                pred_centerness = tf.cast(outputs["centerness"]["level%d" % level], tf.float32)

                h, w = tf.shape(pred_boxes)[1], tf.shape(pred_boxes)[2]
                grid_ys, grid_xs = self.get_grid_points(h, w, 2 ** level)
                grid_ys = tf.cast(grid_ys, pred_boxes.dtype)
                grid_xs = tf.cast(grid_xs, pred_boxes.dtype)
                pred_boxes = tf.stack([grid_xs - pred_boxes[..., 0],
                                       grid_ys - pred_boxes[..., 1],
                                       grid_xs + pred_boxes[..., 2],
                                       grid_ys + pred_boxes[..., 3]], axis=-1)

                pred_boxes = tf.reshape(pred_boxes, [-1, h * w, 4])
                pred_labels = tf.reshape(pred_labels, [-1, h * w, self._label_dims])
                pred_centerness = tf.reshape(pred_centerness, [-1, h * w, 1])
                predicted_boxes_list.append(pred_boxes)
                predicted_labels_list.append(pred_labels)
                predicted_centerness_list.append(pred_centerness)
            
            predicted_boxes = tf.concat(predicted_boxes_list, 1)
            predicted_labels = tf.concat(predicted_labels_list, 1)
            predicted_centerness = tf.concat(predicted_centerness_list, 1)
            
            if self.use_sigmoid:
                predicted_scores = tf.nn.sigmoid(predicted_labels)
            else:
                predicted_scores = tf.nn.softmax(predicted_labels, axis=-1)
                predicted_scores = predicted_scores[:, :, 1:]  

            predicted_centerness = tf.nn.sigmoid(predicted_centerness)
                    
            if "Quality" in self.test_cfg.nms:
                return self.nms(predicted_boxes, predicted_scores, predicted_centerness)
            
            predicted_scores = predicted_scores * predicted_centerness
                
            return self.nms(predicted_boxes, predicted_scores)

 
