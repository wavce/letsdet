import math
import tensorflow as tf
from core import build_loss
from core import build_sampler
from core import build_assigner
from ..head import BaseHead 
from ...builder import HEADS
from core.layers import build_activation

@HEADS.register
class OneNetHead(BaseHead):
    def __init__(self, cfg, **kwargs):
        super(OneNetHead, self).__init__(cfg, **kwargs)
        assert cfg.assigner.assigner == "MinCostAssigner"

        self.feat = tf.keras.layers.Conv2D(
            filters=cfg.feat_dims,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            use_bias=True,
            name="feat1")
    
        self.classifier = tf.keras.layers.Conv2D(
            filters=self.num_classes,
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
        
        self.act = build_activation(**cfg.activation.as_dict())

        self.assigner = build_assigner(**cfg.assigner.as_dict()) if cfg.get("assigner") is not None else None
        self.bbox_loss_func = build_loss(**cfg.bbox_loss.as_dict()) if cfg.get("bbox_loss") is not None else None
        self._use_iou_loss = False
        if self.bbox_loss_func is not None:
            self._use_iou_loss = "IoU" in cfg.bbox_loss.loss
        self.label_loss_func = build_loss(**cfg.label_loss.as_dict()) if cfg.get("label_loss") is not None else None
    
    def get_grid_points(self, feat_height, feat_width, stride):
        with tf.name_scope("get_grid_points"):
            grid_x, grid_y = tf.meshgrid(
                tf.range(0, feat_width), 
                tf.range(0, feat_width))
            grid_y = (tf.cast(grid_y, tf.float32)  + 0.5) * stride
            grid_x = (tf.cast(grid_x, tf.float32)  + 0.5) * stride

            return grid_y, grid_x
    
    def decoder(self, grid_x, grid_y, pred_boxes):
        with tf.name_scope("decoder"):
            pred_boxes = tf.stack([
                grid_x[None] - pred_boxes[..., 0],
                grid_y[None] - pred_boxes[..., 1],
                grid_x[None] + pred_boxes[..., 2],
                grid_y[None] + pred_boxes[..., 3]
            ], -1)
            return pred_boxes

    def call(self, inputs, training=None):
        feat = self.act(self.feat(inputs))

        pred_boxes = self.regressor(feat)
        pred_boxes = tf.nn.relu(pred_boxes)
        feat_h, feat_w = tf.shape(feat)[1], tf.shape(feat)[2]
        grid_y, grid_x = self.get_grid_points(feat_h, feat_w, self.cfg.strides)
        pred_boxes = self.decoder(grid_x, grid_y, pred_boxes)

        pred_labels = self.classifier(feat)

        outputs = dict(boxes=pred_boxes, labels=pred_labels)
        
        if self.is_training:
            return outputs
        
        return self.get_boxes(outputs)

    def get_targets(self, gt_boxes, gt_labels, pred_boxes, pred_labels):
        with tf.name_scope("get_targets"):
            batch_size = tf.shape(gt_boxes)[0]
            target_boxes_ta = tf.TensorArray(tf.float32, batch_size)
            target_labels_ta = tf.TensorArray(tf.int64, batch_size)
            
            for i in tf.range(batch_size):
                t_boxes, t_labels = self.assigner(
                    gt_boxes[i], gt_labels[i], pred_boxes[i], pred_labels[i])

                target_boxes_ta = target_boxes_ta.write(i, t_boxes)
                target_labels_ta = target_labels_ta.write(i, t_labels)
               
            target_boxes = target_boxes_ta.stack()
            target_labels = target_labels_ta.stack()
            
            target_boxes = tf.stop_gradient(target_boxes)
            target_labels = tf.stop_gradient(target_labels)

            return target_boxes, target_labels
    
    def compute_losses(self, predictions, image_info):
        with tf.name_scope("comput_losses"):
            pred_boxes = tf.cast(predictions["boxes"], tf.float32)
            pred_labels = tf.cast(predictions["labels"], tf.float32)
            
            gt_boxes = tf.cast(image_info["boxes"], tf.float32)
            gt_labels = tf.cast(image_info["labels"], tf.int64)
            
            tgt_boxes, tgt_labels = self.get_targets(gt_boxes, gt_labels, pred_boxes, pred_labels)

            float_mask = tf.reduce_max(tgt_labels, -1, keepdims=True)

            bbox_loss = self.bbox_loss_func(tgt_boxes, pred_boxes, float_mask)
            label_loss = self.label_loss_func(tgt_labels, pred_labels)

            num_pos = tf.reduce_sum(float_mask)

            return dict(bbox_loss=bbox_loss / num_pos, label_loss=label_loss / num_pos)


    def get_boxes(self, outputs):
        with tf.name_scope("get_boxes"):
            topk =  self.test_cfg.topk

            boxes = tf.cast(outputs["boxes"], tf.float32)
            bs = tf.shape(boxes)[0]
            labels = tf.nn.sigmoid(tf.cast(outputs["labels"], tf.float32))
            ncls = tf.shape(labels)[-1]
            h, w = tf.shape(labels)[1], tf.shape(labels)[2]
            num_classes = tf.shape(labels)[-1]
            boxes = tf.reshape(boxes, [bs, h * w, 4])
            labels = tf.reshape(labels, [bs, h * w, num_classes])

            scores = tf.transpose(labels, [0, 2, 1])
            topk_scores1, topk_inds1 = tf.nn.top_k(scores, topk)

            topk_scores, topk_inds2 = tf.nn.top_k(tf.reshape(topk_scores1, [bs, -1]), topk)
            
            topk_classes = topk_inds2 // topk
            
            batch_inds = tf.reshape(tf.range(bs), [bs, 1])
            topk_inds = tf.reshape(batch_inds * topk * ncls + topk_inds2, [-1])
            topk_inds = tf.gather(tf.reshape(topk_inds1 + batch_inds[:, :, None] * w * h, [-1]), topk_inds)

            topk_boxes = tf.gather(tf.reshape(boxes, [-1, 4]), topk_inds)
            topk_boxes = tf.reshape(topk_boxes, [bs, -1, 4])
            topk_scores = tf.reshape(topk_scores, [bs, -1])
            topk_classes = tf.reshape(topk_classes, [bs, -1])

            thresh_mask = tf.cast(topk_scores > self.test_cfg.score_threshold, tf.float32)
            valid_detections = tf.cast(tf.reduce_sum(thresh_mask, -1), tf.int32)
            topk_boxes = tf.expand_dims(thresh_mask, -1) * topk_boxes
            topk_score = thresh_mask * topk_scores
            topk_classes = tf.cast(thresh_mask, topk_classes.dtype) * topk_classes

            return dict(nmsed_boxes=topk_boxes,
                        nmsed_scores=topk_score, 
                        nmsed_classes=topk_classes, 
                        valid_detections=valid_detections) 

