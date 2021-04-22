import math
import tensorflow as tf
from ...builder import HEADS
from .anchor_head import AnchorHead 
from core.layers import build_activation
from core.layers import build_convolution
from core.layers import build_normalization


@HEADS.register
class RetinaNetHead(AnchorHead):
    def __init__(self, cfg, anchor_cfg, **kwargs):
        super(RetinaNetHead, self).__init__(cfg, anchor_cfg, **kwargs)
        
        self._make_shared_convs()
        self._make_init_layer()
    
    def _make_init_layer(self):
        self.classifier = build_convolution(self.cfg.convolution,
                                            filters=self._label_dims * self.num_anchors,
                                            kernel_size=(3, 3),
                                            strides=(1, 1),
                                            padding="same",
                                            use_bias=True,
                                            bias_initializer=tf.keras.initializers.Constant(
                                                -math.log((1. - self.cfg.prior) / self.cfg.prior)),
                                            name="class_net/class-predict")
        self.regressor = build_convolution(self.cfg.convolution,
                                           filters=4 * self.num_anchors,
                                           kernel_size=(3, 3),
                                           strides=(1, 1),
                                           padding="same",
                                           use_bias=True,
                                           name="box_net/box-predict")
    
    def _make_shared_convs(self):
        self.box_shared_convs = [
            build_convolution(convolution=self.cfg.convolution,
                              filters=self.cfg.feat_dims,
                              kernel_size=(3, 3),
                              strides=(1, 1),
                              padding="same",
                              use_bias=self.cfg.normalization is None or self.cfg.convolution == "separable_conv2d",
                              name="box_net/box-%d" % i)
            for i in range(self.cfg.repeats)
        ]

        if self.cfg.normalization:
            self.box_norm_layers = {
                "level%d" % level: [
                    build_normalization(
                        **self.cfg.normalization.as_dict(), 
                        name="box_net/box-%d-bn-%d" % (i, level)) 
                    for i in range(self.cfg.repeats)
                ] 
                for level in range(self.min_level, self.max_level + 1)
            }
      
        self.class_shared_convs = [
            build_convolution(convolution=self.cfg.convolution,
                              filters=self.cfg.feat_dims,
                              kernel_size=(3, 3),
                              strides=(1, 1),
                              padding="same",
                              use_bias=self.cfg.normalization is None or self.cfg.convolution == "separable_conv2d",
                              name="class_net/class-%d" % i)
            for i in range(self.cfg.repeats)
        ]
        if self.cfg.normalization:
            self.class_norm_layers = {
                "level%d" % level: [
                    build_normalization(
                        **self.cfg.normalization.as_dict(), 
                        name="class_net/class-%d-bn-%d" % (i, level)) 
                    for i in range(self.cfg.repeats)
                ] 
                for level in range(self.min_level, self.max_level + 1)
            }
        self.act_fn = build_activation(**self.cfg.activation.as_dict())
        
    def call(self, inputs, training=None):
        predicted_boxes = dict()
        predicted_labels = dict()
        total_anchors = dict()
        for i, level in enumerate(range(self.min_level, self.max_level + 1)):
            box_feat = inputs[i]
            label_feat = inputs[i]
            for j in range(self.cfg.repeats):
                box_feat = self.box_shared_convs[j](box_feat)
                if hasattr(self, "box_norm_layers"):
                    box_feat = self.box_norm_layers["level%d" % level][j](box_feat, training=training)
                box_feat = self.act_fn(box_feat)
                
                label_feat = self.class_shared_convs[j](label_feat)
                if hasattr(self, "class_norm_layers"):
                    label_feat = self.class_norm_layers["level%d" % level][j](label_feat, training=training)
                label_feat = self.act_fn(label_feat)
                        
            pred_boxes = self.regressor(box_feat)
            pred_labels = self.classifier(label_feat)

            if self.image_data_format == "channels_first":
                pred_boxes = tf.transpose(pred_boxes, (0, 2, 3, 1), name="box_net/permute_%d" % level)
                pred_labels = tf.transpose(pred_labels, (0, 2, 3, 1), name="class_net/permute_%d" % level)

            feat_h, feat_w = tf.shape(pred_boxes)[1], tf.shape(pred_boxes)[2]
            anchors = self.anchor_generators[i](feat_h, feat_w)
            
            predicted_boxes["level%d" % level] = pred_boxes
            predicted_labels["level%d" % level] = pred_labels
            total_anchors["level%d" % level] = anchors
        
        return predicted_boxes, predicted_labels, total_anchors
    