import math
import tensorflow as tf
from utils import box_utils
from core import build_loss
from ...builder import HEADS
from ...necks.dilated_encoder import make_decoder
from .anchor_head import AnchorHead 
from ...common import ConvNormActBlock
from core import build_anchor_generator 


@HEADS.register
class YOLOFHead(AnchorHead):
    def __init__(self, cfg, **kwargs):
        super(YOLOFHead, self).__init__(cfg=cfg, **kwargs)

        self.cls_subnet, self.reg_subnet = make_decoder(
            feat_dims=cfg.feat_dims,
            cls_num_convs=cfg.cls_num_convs,
            reg_num_convs=cfg.reg_num_convs,
            kernel_size=3,
            data_format=self.data_format,
            kernel_initializer=cfg.kernel_initializer,
            normalization=cfg.normalization.as_dict(),
            activation=cfg.activation.as_dict())
        self._make_init_layers()
        self._init_anchor_generators()
        self.INF = 1e8

    def _init_anchor_generators(self):
        self.anchor_generator = build_anchor_generator(**self.anchor_cfg.as_dict())
     
    def _make_init_layers(self):
        self.classifier = tf.keras.layers.Conv2D(
            filters=self.num_anchors * self.num_classes,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
            bias_initializer=tf.keras.initializers.Constant(-math.log((1. - self.cfg.prior) / self.cfg.prior)),
            name="predicted_class")

        self.regressor = tf.keras.layers.Conv2D(
            filters=self.num_anchors * 4,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            use_bias=True,
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
            name="predicted_box")
        
        self.objectness = tf.keras.layers.Conv2D(
            filters=self.num_anchors,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            use_bias=True,
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
            name="predicted_objecteness")
    
    def call(self, inputs, training=None):
        cls_feat = self.cls_subnet(inputs, training=training)
        reg_feat = self.reg_subnet(inputs, training=training)

        predicted_labels = self.classifier(cls_feat)
        predicted_boxes = self.regressor(reg_feat)
        # implicit objectness
        objectness = self.objectness(reg_feat)

        b = tf.shape(predicted_labels)[0]
        h = tf.shape(predicted_labels)[1]
        w = tf.shape(predicted_labels)[2]

        predicted_labels = tf.reshape(predicted_labels, [b, h, w, self.num_anchors, self.num_classes])
        objectness = tf.reshape(objectness, [b, h, w, self.num_anchors, 1])
        
        tmp = tf.math.log(1. + tf.minimum(tf.exp(predicted_labels), self.INF) + tf.minimum(tf.exp(objectness), self.INF))
        predicted_labels = predicted_labels + objectness - tmp

        predicted_labels = tf.reshape(predicted_labels, [b, h, w, self.num_anchors * self.num_classes])

        anchors = self.anchor_generator(h, w)

        outputs = dict(boxes=predicted_boxes, labels=predicted_labels, total_anchors=anchors)

        if self.is_training:
            return outputs
        
        return self.get_boxes(outputs)
    
    def get_targets(self, gt_boxes, gt_labels, anchors, predicted_boxes):
        batch_size = tf.shape(gt_boxes)[0]
        target_boxes_ta = tf.TensorArray(tf.float32, batch_size)
        target_labels_ta = tf.TensorArray(tf.int64, batch_size)
        label_weights_ta = tf.TensorArray(tf.float32, batch_size)
        box_weights_ta = tf.TensorArray(tf.float32, batch_size)

        for i in tf.range(batch_size):
            t_boxes, t_labels = self.assigner(
                gt_boxes[i], gt_labels[i], anchors, predicted_boxes[i])
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
            predicted_boxes = predictions["boxes"]
            predicted_labels = predictions["labels"]
            total_anchors = predictions["total_anchors"]

            h, w = tf.shape(predicted_boxes)[1], tf.shape(predicted_boxes)[2]
            predicted_boxes = tf.cast(predicted_boxes, tf.float32)
            predicted_labels = tf.cast(predicted_labels, tf.float32)
            predicted_boxes = tf.reshape(predicted_boxes, [-1, h * w * self.num_anchors, 4])
            predicted_labels = tf.reshape(predicted_labels, [-1, h * w * self.num_anchors, self.num_classes])
            total_anchors = tf.cast(total_anchors, tf.float32)
            
            gt_boxes = tf.cast(image_info["boxes"], tf.float32)
            gt_labels = tf.cast(image_info["labels"], tf.int64)

            assert self._use_iou_loss
            predicted_boxes = self.bbox_decoder(total_anchors[None], predicted_boxes)
            
            target_boxes, target_labels, box_weights, label_weights = self.get_targets(
                gt_boxes, gt_labels, total_anchors, predicted_boxes)
            
            if self.use_sigmoid:
                one_hot_target_labels = tf.one_hot(target_labels - 1, self.num_classes)
            else:
                one_hot_target_labels = tf.one_hot(target_labels, self.num_classes + 1)

            label_loss = self.label_loss_func(one_hot_target_labels, predicted_labels, label_weights)
            bbox_loss = self.bbox_loss_func(target_boxes, predicted_boxes, box_weights)
                        
        avg_factor = tf.reduce_sum(box_weights) + tf.cast(tf.shape(box_weights)[0], box_weights.dtype)
        bbox_loss = bbox_loss / avg_factor
        label_loss = label_loss / avg_factor

        return dict(bbox_loss=bbox_loss, label_loss=label_loss)
        
    def get_boxes(self, outputs):
        with tf.name_scope("get_boxes"):
            pred_boxes = tf.cast(outputs["boxes"], tf.float32)
            pred_labels = tf.cast(outputs["labels"], tf.float32)
            anchors = tf.cast(outputs["total_anchors"], tf.float32)
            
            h, w = tf.shape(pred_boxes)[1], tf.shape(pred_boxes)[2]
            pred_boxes = tf.reshape(pred_boxes, [-1, h, w, -1, 4])
            pred_boxes = tf.reshape(pred_boxes, [-1, h * w * self.num_anchors, 4])
            pred_boxes = self.bbox_decoder(anchors[None], pred_boxes)
            
            pred_labels = tf.reshape(pred_labels, [-1, h * w * self.num_anchors, self.num_classes])
            pred_scores = tf.nn.sigmoid(pred_labels)

            return self.nms(pred_boxes, pred_scores)



