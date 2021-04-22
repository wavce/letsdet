import tensorflow as tf

from utils import box_utils
from core import build_loss
from ..head import BaseHead
from ...builder import HEADS
from core import build_sampler
from core import build_assigner
from core.bbox import build_decoder
from core.bbox import build_encoder
from core.layers import build_activation
from core.layers import build_convolution
from core.layers import build_normalization


@HEADS.register
class BBoxHead(tf.keras.Model):
    def __init__(self, 
                 cfg,
                 num_classes=80,
                 num_shared_convs=0,
                 num_shared_fcs=0,
                 num_bbox_fcs=0,
                 num_label_fcs=0,
                 **kwargs):
        super(BBoxHead, self).__init__(**kwargs)

        self.cfg = cfg
        self.num_classes = num_classes

        self._use_iou_loss = "IoU" in cfg.bbox_loss.loss
        self.bbox_loss_func = build_loss(**cfg.bbox_loss.as_dict())
        self.label_loss_func = build_loss(**cfg.label_loss.as_dict())

        self.bbox_decoder = build_decoder(**cfg.bbox_decoder.as_dict())
        self.bbox_encoder = build_encoder(**cfg.bbox_encoder.as_dict())
        
        self.sampler = build_sampler(**cfg.sampler.as_dict())
        self.assigner = build_assigner(**cfg.assigner.as_dict())

        self.use_sigmoid = cfg.use_sigmoid
        self._label_dims = num_classes if cfg.use_sigmoid else num_classes + 1

        self.feat_dims = cfg.feat_dims
        self.roi_feat_size = cfg.roi_feat_size
        self.fc_dims = cfg.fc_dims
        self.conv_dims = cfg.conv_dims
        self.use_avgpool = cfg.use_avgpool

        if num_shared_convs > 0:
            self.shared_convs = []
            for i in range(num_shared_convs):
                self.shared_convs.append(
                    build_convolution(cfg.convolution, 
                                      filters=cfg.conv_dims, 
                                      kernel_size=3, 
                                      padding="same", 
                                      use_bias=cfg.normalization is None, 
                                      name="shared_convs%d" % i))
                if cfg.normalization is not None:
                    self.shared_convs.append(
                        build_normalization(**cfg.normalization.as_dict(), name="shared_convs%d-bn" % i))
                self.shared_convs.append(build_activation(**cfg.activation.as_dict(), name="shared_convs%d-act" % i))
        
        if num_shared_fcs > 0:
            self.shared_fcs = []
            for i in range(num_shared_fcs):
                self.shared_fcs.append(tf.keras.layers.Dense(self.fc_dims, cfg.activation.activation, name="shared_fc%d" % i))
        
        if num_bbox_fcs > 0:
            self.bbox_fcs = []
            for i in range(num_bbox_fcs):
                self.bbox_fcs.append(tf.keras.layers.Dense(self.fc_dims, cfg.activation.activation, name="bbox_fc%d" % i))
        if num_label_fcs > 0:
            self.label_fcs = []
            for i in range(num_label_fcs):
                self.label_fcs.append(tf.keras.layers.Dense(self.fc_dims, cfg.activation.activation, name="label_fc%d" % i))

        self.classifier = tf.keras.layers.Dense(
            units=self._label_dims,
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
            name="classifier")
        self.regressor = tf.keras.layers.Dense(
            units=4 if self.cfg.reg_class_agnostic else 4 * self.num_classes,
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.001),
            name="regressor")
        
        self.use_avgpool = cfg.use_avgpool
    
    @property
    def has_shared_convs(self):
        return hasattr(self, "shared_convs") and self.shared_convs is not None
    
    @property
    def has_shared_fcs(self):
        return hasattr(self, "shared_fcs") and self.shared_fcs is not None

    @property
    def has_bbox_fcs(self):
        return hasattr(self, "bbox_fcs") and self.bbox_fcs is not None
    
    @property
    def has_label_fcs(self):
        return hasattr(self, "label_fcs") and self.label_fcs is not None 

    def call(self, inputs, training=None):
        x = inputs
        if self.has_shared_convs:
            for layer in self.shared_convs:
                if isinstance(layer, tf.keras.layers.BatchNormalization):
                    x = layer(x, training=training)
                else:
                    x = layer(x)
        
        if self.has_shared_fcs:
            if self.use_avgpool:
                x = tf.reduce_mean(x, [2, 3], name="global_avg_pooling", keepdims=True)
            x = tf.reshape(x, [-1, tf.shape(x)[1], self.feat_dims * self.roi_feat_size * self.roi_feat_size])
            for fc in self.shared_fcs:
                x = fc(x)

        box_feats = x
        label_feats = x
        if self.has_bbox_fcs:
            for fc in self.bbox_fcs:
                box_feats = fc(box_feats)
        if self.has_label_fcs:
            for fc in self.label_fcs:
                label_feats = fc(label_feats)

        predicted_boxes = self.regressor(box_feats)
        predicted_labels = self.classifier(label_feats)
        
        return predicted_boxes, predicted_labels
    
    def get_targets(self, rois, gt_boxes, gt_labels):
        with tf.name_scope("get_targets"):
            gt_boxes = tf.cast(gt_boxes, tf.float32)
            gt_labels = tf.cast(gt_labels, tf.int64)
            batch_size = tf.shape(rois)[0]

            rois = tf.cast(rois, tf.float32)
            target_boxes_ta = tf.TensorArray(tf.float32, batch_size, name="target_boxes_ta")
            target_labels_ta = tf.TensorArray(tf.int64, batch_size, name="target_labels_ta")
            box_weights_ta = tf.TensorArray(tf.float32, batch_size, name="box_weights_ta")
            label_weights_ta = tf.TensorArray(tf.float32, batch_size, name="label_weights_ta")
            rois_ta = tf.TensorArray(tf.float32, batch_size, name="rois_ta")

            for i in tf.range(batch_size):                
                t_boxes, t_labels = self.assigner(gt_boxes[i], gt_labels[i], rois[i])
                t_boxes, t_labels, b_weights, l_weights = self.sampler(t_boxes, t_labels, gt_boxes[i], gt_labels[i])
                valid_mask = tf.greater(l_weights, 0)

                t_boxes = tf.boolean_mask(t_boxes, valid_mask)
                t_labels = tf.boolean_mask(t_labels, valid_mask)
                l_weights = tf.boolean_mask(l_weights, valid_mask)
                b_weights = tf.boolean_mask(b_weights, valid_mask)
                valid_rois = tf.boolean_mask(tf.concat([gt_boxes[i], rois[i]], 0), valid_mask)

                box_weights_ta = box_weights_ta.write(i, b_weights)
                label_weights_ta = label_weights_ta.write(i, l_weights)
                target_boxes_ta = target_boxes_ta.write(i, t_boxes)
                target_labels_ta = target_labels_ta.write(i, t_labels)
                rois_ta = rois_ta.write(i, valid_rois)

            rois = rois_ta.stack(name="rois")
            target_boxes = target_boxes_ta.stack(name="target_boxes")
            target_labels = target_labels_ta.stack(name="target_labels")
            box_weights = box_weights_ta.stack(name="box_weights")
            label_weights = label_weights_ta.stack(name="label_weights")

            return rois, target_boxes, target_labels, box_weights, label_weights
   
    def compute_losses(self, predicted_boxes, predicted_labels, rois, target_boxes, target_labels, box_weights, label_weights):
        with tf.name_scope("compute_losses"):
            predicted_boxes = tf.cast(predicted_boxes, tf.float32)
            predicted_labels = tf.cast(predicted_labels, tf.float32)
            rois = tf.cast(rois, tf.float32)

            target_labels = tf.one_hot(target_labels, self._label_dims)
            
            if self._use_iou_loss:
                predicted_boxes = self.bbox_decoder(rois, predicted_boxes)
            else:
                target_boxes = self.bbox_encoder(rois, target_boxes)
            
            label_loss = self.label_loss_func(target_labels, predicted_labels, label_weights)
            bbox_loss = self.bbox_loss_func(target_boxes, predicted_boxes, box_weights) 
                        
            label_loss = tf.reduce_sum(label_loss) / (tf.reduce_sum(box_weights) + 1.)
            bbox_loss = tf.reduce_sum(bbox_loss)  / (tf.reduce_sum(box_weights) + 1.)

            return dict(rcnn_bbox_loss=bbox_loss, rcnn_label_loss=label_loss)


@HEADS.register
class Shared2FCBBoxHead(BBoxHead):
    def __init__(self, **kwargs):
        super(Shared2FCBBoxHead, self).__init__(num_shared_convs=0, 
                                                num_shared_fcs=2,
                                                num_bbox_fcs=0,
                                                num_label_fcs=0,
                                                **kwargs)


@HEADS.register
class Shared4Conv1FCBBoxHead(BBoxHead):
    def __init__(self, **kwargs):
        super(Shared4Conv1FCBBoxHead, self).__init__(num_shared_convs=4, 
                                                     num_shared_fcs=1,
                                                     num_bbox_fcs=0,
                                                     num_label_fcs=0,
                                                     **kwargs)
