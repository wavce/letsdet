import numpy as np
import tensorflow as tf 
from . import Detector
from ..builder import DETECTORS
from ..builder import build_head
from ..builder import build_neck
from ..builder import build_backbone


class OneStageDetector(Detector):
    def __init__(self, cfg, training=True):
        super(OneStageDetector, self).__init__(cfg, training=training)

        self.data_format = cfg.data_format

        inputs = tf.keras.Input(shape=(None, None, 3))
        self.backbone = build_backbone(input_tensor=inputs, **cfg.backbone.as_dict())
        x = self.backbone(inputs)
        
        if cfg.get("neck"):
            if isinstance(x, (list, tuple)):
                input_shapes = [i.shape.as_list()[1:] for i in x]
                if cfg.neck.get("downsample_ratio"):  
                    first_level = int(np.log2(cfg.neck.downsample_ratio))   ## for centernet
                    x = x[first_level:]
            else:
                input_shapes = x.shape.as_list()[1:]
            self.neck = build_neck(input_shapes=input_shapes, name="neck", **cfg.neck.as_dict())
            x = self.neck(x)

        if cfg.get("anchors"):
            self.head = build_head(cfg.head.head, 
                                   cfg=cfg.head, 
                                   test_cfg=cfg.test,
                                   anchor_cfg=cfg.anchors, 
                                   num_classes=cfg.num_classes,
                                   is_training=training, 
                                   name="head")
        else:
            self.head = build_head(cfg.head.head, 
                                   cfg=cfg.head,
                                   test_cfg=cfg.test, 
                                   num_classes=cfg.num_classes, 
                                   is_training=training,
                                   name="head")
        x = self.head(x)
        self.detector = tf.keras.Model(inputs=inputs, outputs=x)
    
    def load_pretrained_weights(self, pretrained_weights_path=None):
        if pretrained_weights_path:
            self.backbone.load_weights(pretrained_weights_path, by_name=True, skip_mismatch=True)
            print("Restored pre-trained weights from %s." % pretrained_weights_path)
        
        else:
            print("Train model from scratch.")
    
    def compute_losses(self, predictions, image_info):
        return self.head.compute_losses(predictions, image_info)
    
    def save_weights(self, name):
        self.detector.save_weights(name)
    
    @tf.function
    def __call__(self, inputs, training):
        x = self.detector(inputs, training=training)
        return x

