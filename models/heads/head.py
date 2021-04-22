import tensorflow as tf
from utils import box_utils
from core import build_loss
from core import build_sampler
from core import build_assigner
from core.builder import build_nms
from core.bbox import build_decoder
from core.bbox import build_encoder
from ..common import ConvNormActBlock 


class BaseHead(tf.keras.Model):
    def __init__(self, cfg, test_cfg, anchor_cfg=None, num_classes=80, is_training=True, data_format="channels_last", **kwargs):
        super(BaseHead, self).__init__(**kwargs)
      
        self.num_classes = num_classes
        self.cfg = cfg
        self.anchor_cfg = anchor_cfg
        self.test_cfg = test_cfg
        self.is_training = is_training
        self.data_format = data_format

        if test_cfg and test_cfg.get("nms") is not None:
            self.nms = build_nms(**test_cfg.as_dict()) 
       
        self.use_sigmoid = True
        if cfg.get("use_sigmoid") is not None:
            self.use_sigmoid = cfg.use_sigmoid 
        self._label_dims = num_classes if self.use_sigmoid else num_classes + 1

        self.bbox_loss_func = build_loss(**cfg.bbox_loss.as_dict()) if cfg.get("bbox_loss") is not None else None
        self._use_iou_loss = False
        if self.bbox_loss_func is not None:
            self._use_iou_loss = "IoU" in cfg.bbox_loss.loss
        self.label_loss_func = build_loss(**cfg.label_loss.as_dict()) if cfg.get("label_loss") is not None else None

        self.sampler = build_sampler(**cfg.sampler.as_dict()) if cfg.get("sampler") is not None else None
        self.assigner = build_assigner(**cfg.assigner.as_dict()) if cfg.get("assigner") is not None else None

        self.bbox_decoder = build_decoder(**cfg.bbox_decoder.as_dict()) if cfg.get("bbox_decoder") is not None else None
        self.bbox_encoder = build_encoder(**cfg.bbox_encoder.as_dict()) if cfg.get("bbox_encoder") is not None else None
          
    @property
    def min_level(self):
        if self.cfg.get("min_level"):
            return self.cfg.min_level
        
        return None
    
    @property
    def max_level(self):
        if self.cfg.get("max_level"):
            return self.cfg.max_level
        return None
         
    def _make_shared_convs(self):
        self.box_shared_convs = tf.keras.Sequential(name="box_net")
        self.class_shared_convs = tf.keras.Sequential(name="cls_net")

        for i in range(self.cfg.repeats):
            self.box_shared_convs.add(
                ConvNormActBlock(filters=self.cfg.feat_dims,
                                 kernel_size=(3, 3),
                                 padding="same",
                                 strides=(1, 1),
                                 normalization=self.cfg.normalization.as_dict() if self.cfg.normalization else None,
                                 activation=self.cfg.activation.as_dict(),
                                 name="%d" % i))
            self.class_shared_convs.add(
                ConvNormActBlock(filters=self.cfg.feat_dims,
                                 kernel_size=(3, 3),
                                 strides=(1, 1),
                                 padding="same",
                                 normalization=self.cfg.normalization.as_dict() if self.cfg.normalization else None,
                                 activation=self.cfg.activation.as_dict(),
                                 name="%d" % i))
                   
    def get_targets(self, gt_boxes, gt_labels, total_anchors):
        raise NotImplementedError()

    def compute_losses(self, predictions, image_info):
        raise NotImplementedError()

    def get_boxes(self, outputs):
        raise NotImplementedError()

    