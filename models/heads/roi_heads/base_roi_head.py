import tensorflow as tf
from ..head import BaseHead


class BaseRoIHead(BaseHead):
    def __init__(self, cfg, test_cfg, num_classes=80, is_training=True, **kwargs):
        super(BaseRoIHead, self).__init__(cfg=cfg, test_cfg=test_cfg, num_classes=num_classes, is_training=is_training, **kwargs)
        
        if cfg.get("bbox_head"):
            self.pooled_size = cfg.bbox_head.roi_pooling.pooled_size
            self._make_bbox_head(cfg.bbox_head)
        
        if cfg.get("mask_head"):
            self._make_mask_head(cfg.mask_head)
    
    @property
    def min_level(self):
        return self.cfg.get("min_level")
    
    @property
    def max_level(self):
        return self.cfg.get("max_level")
    
    def _make_bbox_head(self, bbox_head_cfg):
        raise NotImplementedError()

    def _make_mask_head(self, mask_head_cfg):
        raise NotImplementedError()

    @property
    def has_bbox_head(self):
        return hasattr(self, "bbox_head")
    
    @property
    def has_mask_head(self):
        return hasattr(self, "mask_head")
    
    