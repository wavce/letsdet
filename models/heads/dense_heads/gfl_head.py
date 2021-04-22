import math
import tensorflow as tf
from core import build_loss
from ...builder import HEADS
from core.layers import Scale
from core.bbox import compute_iou
from .anchor_head import AnchorHead 


@HEADS.register
class GFLHead(AnchorHead):
    def __init__(self, cfg, anchor_cfg, num_classes, **kwargs):
        super(GFLHead, self).__init__(cfg=cfg, anchor_cfg=anchor_cfg, num_classes=num_classes, **kwargs)

        self._make_shared_convs()
        self._init_anchor_generators()

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
            filters=4 * (cfg.reg_max + 1),
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            use_bias=True,
            name="predicted_box")
        self.scales = [
            Scale(value=1.0, name="scales/%d" % i) 
            for i, _ in enumerate(range(cfg.min_level, cfg.max_level + 1))
        ]

        self._num_anchors_per_level = dict()
        self.project = tf.linspace(0, cfg.reg_max, cfg.reg_max + 1)
        self.dfl_loss_func = build_loss(**cfg.dfl_loss.as_dict())
    
    def call(self, inputs, training=None):
        predicted_boxes = dict()
        predicted_labels = dict()
        total_anchors = dict()

        for i, level in enumerate(range(self.min_level, self.max_level + 1)):
            box_feat = self.box_shared_convs(inputs[i], training=training)
            label_feat = self.class_shared_convs(inputs[i], training=training)

            h, w = tf.shape(box_feat)[1], tf.shape(box_feat)[2]

            pred_boxes = self.regressor(box_feat)
            pred_boxes = self.scales[i](pred_boxes)

            pred_labels = self.classifier(label_feat)

            anchors = self.anchor_generators[i](h, w)

            predicted_boxes["level%d" % level] = pred_boxes
            predicted_labels["level%d" % level] = pred_labels
            total_anchors["level%d" % level] = anchors

        outputs = dict(boxes=predicted_boxes, 
                       labels=predicted_labels, 
                       total_anchors=total_anchors)
        if self.is_training:
            return outputs

        return self.get_boxes(outputs)

    def integral(self, inputs):
        with tf.name_scope("integral"):
            x = tf.nn.softmax(inputs, -1)
            project = tf.cast(self.project, x.dtype)
            x = tf.reduce_sum(x * project, -1)

            return x
    
    def get_targets(self, gt_boxes, gt_labels, total_anchors, num_anchors_per_level):
        batch_size = tf.shape(gt_boxes)[0]
        target_boxes_ta = tf.TensorArray(tf.float32, batch_size)
        target_labels_ta = tf.TensorArray(tf.int64, batch_size)
        label_weights_ta = tf.TensorArray(tf.float32, batch_size)
        box_weights_ta = tf.TensorArray(tf.float32, batch_size)

        for i in tf.range(batch_size):
            t_boxes, t_labels = self.assigner(
                gt_boxes[i], gt_labels[i], total_anchors, list(num_anchors_per_level))
            t_boxes, t_labels, b_weights, l_weights = self.sampler(t_boxes, t_labels)

            target_boxes_ta = target_boxes_ta.write(i, t_boxes)
            target_labels_ta = target_labels_ta.write(i, t_labels)
            box_weights_ta = box_weights_ta.write(i, b_weights)
            label_weights_ta = label_weights_ta.write(i, l_weights)
        
        target_boxes = target_boxes_ta.stack()
        target_labels = target_labels_ta.stack()
        box_weights = box_weights_ta.stack()
        label_weights = label_weights_ta.stack()

        target_boxes = tf.stop_gradient(target_boxes)
        target_labels = tf.stop_gradient(target_labels)
        box_weights = tf.stop_gradient(box_weights)
        label_weights = tf.stop_gradient(label_weights)

        return target_boxes, target_labels, box_weights, label_weights

    def compute_losses(self, predictions, image_info):
        with tf.name_scope("compute_losses"):
            bbox_loss_list = []
            qfl_loss_list = []
            dfl_loss_list = []
            pos_avg_factor_list = []

            predicted_boxes = predictions["boxes"]
            predicted_labels = predictions["labels"]
            total_anchors = predictions["total_anchors"]
            
            gt_boxes = tf.cast(image_info["boxes"], tf.float32)
            gt_labels = tf.cast(image_info["labels"], tf.int64)
            
            num_anchors_per_level = [tf.keras.backend.int_shape(a)[0] for _, a in total_anchors.items()]
            
            concat_anchors = tf.concat([v for _, v in total_anchors.items()], 0)
            target_boxes, target_labels, box_weights, label_weights = self.get_targets(
                gt_boxes, gt_labels, concat_anchors, num_anchors_per_level)
            
            if self.use_sigmoid:
                one_hot_target_labels = tf.one_hot(target_labels - 1, self.num_classes)
            else:
                one_hot_target_labels = tf.one_hot(target_labels, self.num_classes + 1)

            start_ind = 0
            for i, level in enumerate(range(self.min_level, self.max_level+1)):
                pred_boxes = tf.cast(predicted_boxes["level%d" % level], tf.float32)
                pred_labels = tf.cast(predicted_labels["level%d" % level], tf.float32)
                anchors = tf.cast(total_anchors["level%d" % level], tf.float32)

                anchors_xs = (anchors[:, 0] + anchors[:, 2]) * 0.5 / (2 ** level)
                anchors_ys = (anchors[:, 1] + anchors[:, 3]) * 0.5 / (2 ** level)

                end_ind = start_ind + num_anchors_per_level[i]
                tgt_boxes = target_boxes[:, start_ind:end_ind, :] / (2 ** level)
                tgt_dist = self.bbox_encoder(tgt_boxes, anchors_ys, anchors_xs)
                tgt_dist = tf.clip_by_value(tgt_dist, 0, self.cfg.reg_max - 0.1)
                tgt_labels = one_hot_target_labels[:, start_ind:end_ind, :]
                
                h, w = tf.shape(pred_boxes)[1], tf.shape(pred_boxes)[2]
            
                pred_boxes = tf.reshape(pred_boxes, [-1, h * w * self.num_anchors, 4, self.cfg.reg_max + 1])
                pred_labels = tf.reshape(pred_labels, [-1, h * w * self.num_anchors, self._label_dims])

                integral_pred_boxes = self.integral(pred_boxes)   
                decoded_pred_boxes = self.bbox_decoder(integral_pred_boxes, anchors_xs, anchors_ys)
                iou = compute_iou(tgt_boxes, decoded_pred_boxes)

                target_weights = tf.reduce_max(tf.nn.sigmoid(pred_labels), -1)
                target_weights = tf.stop_gradient(target_weights) * box_weights[:, start_ind:end_ind]
                qfl_loss = self.label_loss_func((tgt_labels, iou), pred_labels, label_weights[:, start_ind:end_ind])
                dfl_loss = self.dfl_loss_func(tf.reshape(tgt_dist, [-1]), 
                                              tf.reshape(pred_boxes, [-1, self.cfg.reg_max + 1]), 
                                              tf.reshape(tf.tile(tf.expand_dims(target_weights, -1), [1, 1, 4]), [-1]))
                bbox_loss = self.bbox_loss_func(tgt_boxes, decoded_pred_boxes, target_weights)
               
                bbox_loss_list.append(bbox_loss)
                qfl_loss_list.append(qfl_loss)
                dfl_loss_list.append(dfl_loss)
                pos_avg_factor_list.append(tf.reduce_sum(target_weights))
                start_ind = end_ind
            
            avg_factor = tf.reduce_sum(box_weights) + tf.cast(tf.shape(box_weights)[0], box_weights.dtype)
            pos_avg_factor = tf.add_n(pos_avg_factor_list)
            bbox_loss = tf.add_n(bbox_loss_list) / pos_avg_factor
            qfl_loss = tf.add_n(qfl_loss_list) / avg_factor
            dfl_loss = tf.add_n(dfl_loss_list) / pos_avg_factor / 4

            return dict(bbox_loss=bbox_loss, qfl_loss=qfl_loss, dfl_loss=dfl_loss)
    
    def get_boxes(self, outputs):
        with  tf.name_scope("get_boxes"):
            predicted_boxes_list = []
            predicted_labels_list = []
            for _, level in enumerate(range(self.min_level, self.max_level+1)):
                pred_boxes = tf.cast(outputs["boxes"]["level%d" % level], tf.float32)
                pred_labels = tf.cast(outputs["labels"]["level%d" % level], tf.float32)
                anchors = tf.cast(outputs["total_anchors"]["level%d" % level], tf.float32)
                
                h, w = tf.shape(pred_boxes)[1], tf.shape(pred_boxes)[2]
                pred_boxes = tf.reshape(pred_boxes, [-1, h * w, 4, self.cfg.reg_max + 1])
                pred_boxes = self.integral(pred_boxes) * (2 ** level)
                anchors_xs = (anchors[:, 0] + anchors[:, 2]) * 0.5 
                anchors_ys = (anchors[:, 1] + anchors[:, 3]) * 0.5

                pred_boxes = self.bbox_decoder(pred_boxes, anchors_xs, anchors_ys)

                pred_boxes = tf.reshape(pred_boxes, [-1, h * w, 4])
                pred_labels = tf.reshape(pred_labels, [-1, h * w, self._label_dims])
               
                predicted_boxes_list.append(pred_boxes)
                predicted_labels_list.append(pred_labels)
            
            predicted_boxes = tf.concat(predicted_boxes_list, 1)
            predicted_labels = tf.concat(predicted_labels_list, 1)
        
            if self.use_sigmoid:
                predicted_scores = tf.nn.sigmoid(predicted_labels)
            else:
                predicted_scores = tf.nn.softmax(predicted_labels, axis=-1)
                predicted_scores = predicted_scores[:, :, 1:]  

            return self.nms(predicted_boxes, predicted_scores)
 

