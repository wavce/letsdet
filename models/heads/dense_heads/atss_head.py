import math
import numpy as np
import tensorflow as tf
from utils import box_utils
from core import build_loss
from ...builder import HEADS
from core.layers import Scale
from .anchor_head import AnchorHead 
from core.layers import build_activation
from core.layers import build_convolution
from core.layers import build_normalization


@HEADS.register
class ATSSHead(AnchorHead):
    def __init__(self, cfg, test_cfg, anchor_cfg, num_classes, **kwargs):
        super(ATSSHead, self).__init__(cfg=cfg, 
                                       test_cfg=test_cfg, 
                                       anchor_cfg=anchor_cfg, 
                                       num_classes=num_classes, 
                                       **kwargs)
    
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
            Scale(value=1.0, name="scales/%d" % i) 
            for i, _ in enumerate(range(cfg.min_level, cfg.max_level + 1))
        ]
        
        self.centerness_loss_func = build_loss(**self.cfg.centerness_loss.as_dict())

    def call(self, inputs, training=None):
        predicted_boxes = dict()
        predicted_labels = dict()
        predicted_centerness = dict()
        total_anchors = dict()

        for i, level in enumerate(range(self.min_level, self.max_level+1)):
            box_feat = self.box_shared_convs(inputs[i], training=training)
            label_feat = self.class_shared_convs(inputs[i], training=training)
            
            pred_boxes = self.regressor(box_feat)
            
            pred_boxes = self.scales[i](pred_boxes)

            pred_labels = self.classifier(label_feat)
            
            pred_centerness = self.centerness(box_feat)

            h, w = tf.shape(pred_boxes)[1], tf.shape(pred_boxes)[2]
            anchors = self.anchor_generators[i](h, w)

            predicted_boxes["level%d" % level] = pred_boxes
            predicted_labels["level%d" % level] = pred_labels
            predicted_centerness["level%d" % level] = pred_centerness
            total_anchors["level%d" % level] = anchors

        outputs = dict(boxes=predicted_boxes, 
                       labels=predicted_labels, 
                       centerness=predicted_centerness, 
                       total_anchors=total_anchors)
        if self.is_training:
            return outputs
        
        return self.get_boxes(outputs)
    
    def get_targets(self, gt_boxes, gt_labels, total_anchors, num_anchors_per_level):
        batch_size = tf.shape(gt_boxes)[0]
        target_boxes_ta = tf.TensorArray(tf.float32, batch_size)
        target_labels_ta = tf.TensorArray(tf.int64, batch_size)
        target_centerness_ta = tf.TensorArray(tf.float32, batch_size)
        label_weights_ta = tf.TensorArray(tf.float32, batch_size)
        box_weights_ta = tf.TensorArray(tf.float32, batch_size)

        for i in tf.range(batch_size):
            t_boxes, t_labels = self.assigner(
                gt_boxes[i], gt_labels[i], total_anchors, num_anchors_per_level)
            t_centerness = self.assigner.compute_centerness(t_boxes, total_anchors)
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

        return target_boxes, target_labels, target_centerness, box_weights, label_weights

    def compute_losses(self, predictions, image_info):
        with tf.name_scope("compute_losses"):
            bbox_loss_list = []
            label_loss_list = []
            centerness_loss_list = []
            predicted_boxes = predictions["boxes"]
            predicted_labels = predictions["labels"]
            predicted_centerness = predictions["centerness"]
            total_anchors = predictions["total_anchors"]

            gt_boxes = tf.cast(image_info["boxes"], tf.float32)
            gt_labels = tf.cast(image_info["labels"], tf.int64)

            concat_anchors = tf.concat([v for _, v in total_anchors.items()], 0)
            num_anchors_per_level = [tf.keras.backend.int_shape(v)[0] for _, v in total_anchors.items()]
            
            target_boxes, target_labels, target_centerness, box_weights, label_weights = self.get_targets(
                gt_boxes, gt_labels, concat_anchors, num_anchors_per_level)
            
            num_anchors_per_level = [tf.keras.backend.int_shape(a)[0] for _, a in total_anchors.items()]
            if self.use_sigmoid:
                one_hot_target_labels = tf.one_hot(target_labels - 1, self.num_classes)
            else:
                one_hot_target_labels = tf.one_hot(target_labels, self.num_classes + 1)
            
            start_ind = 0
            for i, level in enumerate(range(self.min_level, self.max_level+1)):
                pred_boxes = tf.cast(predicted_boxes["level%d" % level], tf.float32)
                pred_labels = tf.cast(predicted_labels["level%d" % level], tf.float32)
                pred_centerness = tf.cast(predicted_centerness["level%d" % level], tf.float32)
                anchors = tf.cast(total_anchors["level%d" % level], tf.float32)

                h, w = tf.shape(pred_boxes)[1], tf.shape(pred_boxes)[2]
            
                pred_boxes = tf.reshape(pred_boxes, [-1, h * w * self.num_anchors, 4])
                pred_labels = tf.reshape(pred_labels, [-1, h * w * self.num_anchors, self._label_dims])
                pred_centerness = tf.reshape(pred_centerness, [-1, h * w * self.num_anchors, 1])

                end_ind = start_ind + num_anchors_per_level[i]
                tgt_boxes = target_boxes[:, start_ind:end_ind, :]
                tgt_labels = one_hot_target_labels[:, start_ind:end_ind, :]
                tgt_centerness = target_centerness[:, start_ind:end_ind, :]
                
                if self._use_iou_loss:
                    pred_boxes = self.bbox_decoder(anchors[None], pred_boxes)
                else:
                    tgt_boxes = self.bbox_encoder(anchors[None], tgt_boxes)

                label_loss = self.label_loss_func(tgt_labels, pred_labels, label_weights[:, start_ind:end_ind])
                tgt_centerness = tgt_centerness * tf.expand_dims(box_weights[:, start_ind:end_ind], -1)
                bbox_loss = self.bbox_loss_func(tgt_boxes, pred_boxes, tgt_centerness)
                centerness_loss = self.centerness_loss_func(tgt_centerness, pred_centerness, box_weights[:, start_ind:end_ind])
             
                bbox_loss_list.append(bbox_loss)
                label_loss_list.append(label_loss)
                centerness_loss_list.append(centerness_loss)
                start_ind = end_ind
            
            avg_factor = tf.reduce_sum(box_weights) + tf.cast(tf.shape(box_weights)[0], box_weights.dtype)
            bbox_avg_factor = (tf.reduce_sum(box_weights * tf.squeeze(target_centerness, -1)) + 
                               tf.cast(tf.shape(box_weights)[0], box_weights.dtype))
   
            bbox_loss = tf.add_n(bbox_loss_list) / bbox_avg_factor
            label_loss = tf.add_n(label_loss_list) / avg_factor
            centerness_loss = tf.add_n(centerness_loss_list) / avg_factor

            return dict(bbox_loss=bbox_loss, label_loss=label_loss, centerness_loss=centerness_loss)
                
    def get_boxes(self, outputs):
        with  tf.name_scope("get_boxes"):
            predicted_boxes_list = []
            predicted_labels_list = []
            predicted_centerness_list = []
            for _, level in enumerate(range(self.min_level, self.max_level+1)):
                pred_boxes = tf.cast(outputs["boxes"]["level%d" % level], tf.float32)
                pred_labels = tf.cast(outputs["labels"]["level%d" % level], tf.float32)
                pred_centerness = tf.cast(outputs["centerness"]["level%d" % level], tf.float32)

                anchors = tf.cast(outputs["total_anchors"]["level%d" % level], tf.float32)

                h, w = tf.shape(pred_boxes)[1], tf.shape(pred_boxes)[2]

                pred_boxes = tf.reshape(pred_boxes, [-1, h * w, 4])
                pred_labels = tf.reshape(pred_labels, [-1, h * w, self._label_dims])
                pred_centerness = tf.reshape(pred_centerness, [-1, h * w, 1])
                
                pred_boxes = self.bbox_decoder(anchors[None], pred_boxes)
               
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


   