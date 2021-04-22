import math
import tensorflow as tf
from ...builder import HEADS
from .anchor_head import AnchorHead 
from core.layers import build_activation
from core.layers import build_convolution
from core.layers import build_normalization


@HEADS.register
class RetinaNetHead(AnchorHead):
    def __init__(self, **kwargs):
        super(RetinaNetHead, self).__init__(**kwargs)
        
        self._make_shared_convs()
        self._make_init_layers()
        self._init_anchor_generators()
    
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
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
            name="predicted_box")
        
    def call(self, inputs, training=None):
        predicted_boxes = dict()
        predicted_labels = dict()
        total_anchors = dict()
        for i, level in enumerate(range(self.min_level, self.max_level + 1)):
            box_feat = self.box_shared_convs(inputs[i], training=training)
            label_feat = self.class_shared_convs(inputs[i], training=training)

            pred_boxes = self.regressor(box_feat)
            pred_labels = self.classifier(label_feat)

            h, w = tf.shape(box_feat)[1], tf.shape(box_feat)[2]
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
    