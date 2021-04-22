import tensorflow as tf 
from . import Detector
from utils import box_utils
from ..builder import build_neck
from ..builder import build_head
from core.bbox import build_decoder
from ..builder import build_backbone
from core.layers import ProposalLayer


class TwoStageDetector(Detector):
    def __init__(self, cfg, training=True, **kwargs):
        super(TwoStageDetector, self).__init__(cfg, **kwargs)

        self.data_format = cfg.data_format

        inputs = tf.keras.Input(shape=(None, None, 3))
        self.backbone = build_backbone(input_tensor=inputs, **cfg.backbone.as_dict())
        x = self.backbone(inputs)
        
        if cfg.get("neck"):
            if isinstance(x, (list, tuple)):
                input_shapes = [i.shape.as_list()[1:] for i in x]
            else:
                input_shapes = x.shape.as_list()[1:]
            self.neck = build_neck(input_shapes=input_shapes, name="neck", **cfg.neck.as_dict())
            x = self.neck(x)

        if cfg.get("anchors"):
            self.rpn_head = build_head(cfg.rpn_head.head, 
                                       cfg=cfg.rpn_head, 
                                       anchor_cfg=cfg.anchors, 
                                       is_training=training, 
                                       name="rpn_head")
        else:
            self.rpn_head = build_head(cfg.rpn_head.head, 
                                       cfg=cfg.rpn_head,                                       
                                       is_training=training,
                                       name="rpn_head")
        rpn_ouputs, proposals = self.rpn_head(x)
        x = build_head(cfg.roi_head.head, 
                       cfg=cfg.roi_head, 
                       test_cfg=cfg.test, 
                       num_classes=cfg.num_classes,  
                       is_training=training,
                       name="roi_heads")([x, proposals])
        
        self.detector = tf.keras.Model(inputs=inputs, outputs=[proposals, x])
    
    def load_pretrained_weights(self, pretrained_weights_path=None):
        if pretrained_weights_path:
            self.backbone.load_weights(pretrained_weights_path, by_name=True, skip_mismatch=True)
            print("Restored pre-trained weights from %s." % pretrained_weights_path)
        
        else:
            print("Train model from scratch.")
    
    def compute_losses(self, rpn_outputs, rcnn_ouputs, image_info):
        return self.rpn_head.compute_losses(rpn_outputs, image_info)
    
    def save_weights(self, name):
        self.detector.save_weights(name)
    
    @tf.function
    def __call__(self, inputs, training):
        x = self.detector(inputs, training=training)
        return x



