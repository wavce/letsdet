 
 import tensorflow as tf 
from utils import box_utils
from ..builder import DETECTORS
from ..builder import build_neck
from ..builder import build_head
from core.bbox import build_decoder
from ..builder import build_backbone
from core.layers import ProposalLayer
from .two_stage import TwoStageDetecor 


@DETECTORS.register
class CascadeRCNN(TwoStageDetecor):
    def __init__(self, cfg, proposal_cfg, return_loss=True, **kwargs):
        super(CascadeRCNN, self).__init__(cfg, proposal_cfg, return_loss, **kwargs)
    
    def _make_rcnn_head(self, bbox_head_cfg, roi_pooling_cfg):
        self.bbox_roi_pooling = build_roi_pooling(roi_pooling_cfg.roi_pooling, 
                                                  cropped_size=roi_pooling_cfg.cropped_size,
                                                  strides=roi_pooling_cfg.strides)
        self.bbox_head = build_head(bbox_head_cfg.head, cfg=bbox_head_cfg)

    def _make_mask_head(self, mask_head_cfg, mask_roi_pooling_cfg):
        if mask_roi_pooling_cfg is not None:
            self.mask_roi_pooling = build_roi_pooling(mask_roi_pooling_cfg)
            self.share_roi_pooling = False
        else:
            self.share_roi_pooling = True
            self.mask_roi_pooling = self.bbox_roi_pooling
        
        self.mask_head = build_head(mask_head_cfg.head, cfg=mask_head_cfg)
    
    def _make_proposal_layer(self, proposal_cfg):
        pass
    
    @tf.function(experimental_relax_shapes=True) 
    def call(self, inputs, training=None):
        images, image_info = inputs
        
        images = tf.image.convert_image_dtype(images, tf.float32)
        backbone_outputs = self.backbone(images, training=training)

        neck_outputs = self.neck(backbone_outputs, training=training)

        if self.return_loss:
            rpn_boxes, rpn_labels, total_anchors, rpn_box_loss, rpn_label_loss = self.rpn_head(
                [neck_outputs, image_info], training=training)   

            rois, rois_scores = self.get_proposals(rpn_boxes, rpn_labels, total_anchors, image_info) 
            if training:
                rois = tf.stop_gradient(rois)
                rois_scores = tf.stop_gradient(rois_scores)

            rcnn_boxes, rcnn_labels, rois, rcnn_box_loss, rcnn_label_loss = self.rcnn_head(
                [neck_outputs, rois, image_info], training=training)
            
            losses = dict(rpn_box_loss=rpn_box_loss, 
                          rpn_label_loss=rpn_label_loss,
                          rcnn_box_loss=rcnn_box_loss,
                          rcnn_label_loss=rcnn_label_loss)
            outputs = dict(rpn_boxes=rpn_boxes, 
                           rpn_labels=rpn_labels, 
                           total_anchors=total_anchors,
                           rcnn_boxes=rcnn_boxes,
                           rcnn_labels=rcnn_labels,
                           rois=rois)

            return outputs, losses

        rpn_boxes, rpn_labels, total_anchors = self.rpn_head(
            neck_outputs, image_info=image_info, training=training)   
        rois, rois_scores = self.get_proposals(
            rpn_boxes, rpn_labels, total_anchors, image_info) 

        rcnn_boxes, rcnn_labels, rois = self.rcnn_head(neck_outputs, rois, image_info, training=training)

        outputs = dict(rpn_boxes=rpn_boxes, 
                       rpn_labels=rpn_labels, 
                       total_anchors=total_anchors,
                       rcnn_boxes=rcnn_boxes, 
                       rcnn_labels=rcnn_labels,
                       rois=rois)

        return self.get_boxes(outputs, image_info)
    
    def get_proposals(self, rpn_boxes, rpn_labels, anchors, image_info):
        with tf.name_scope("proposals_layer"):
            height = tf.cast(image_info["input_size"][:, 0:1, None], tf.float32)
            width = tf.cast(image_info["input_size"][:, 1:2, None], tf.float32)
            valid_height = tf.cast(image_info["valid_size"][:, 0:1, None], tf.float32)
            valid_width = tf.cast(image_info["valid_size"][:, 1:2, None], tf.float32)

            rpn_boxes = tf.cast(rpn_boxes, tf.float32)
            rpn_labels = tf.cast(rpn_labels, tf.float32)
            anchors = tf.cast(anchors, tf.float32)
            
            if self.rpn_head.use_sigmoid:
                rpn_scores = tf.nn.sigmoid(rpn_labels)
            else:
                rpn_scores =  tf.nn.softmax(rpn_labels, -1)[:, :, 1:]

            rpn_boxes = self.rpn_head.bbox_decoder(anchors, rpn_boxes)
            rpn_boxes = box_utils.clip_boxes(rpn_boxes, valid_height, valid_width)
            rpn_boxes, rpn_scores = box_utils.filter_boxes(
                rpn_boxes, rpn_scores, self.proposal_cfg.min_size, valid_height, valid_width)
            rpn_boxes = box_utils.to_normalized_coordinates(rpn_boxes, height, width)
            
            rois, rois_scores = self.proposal_layer(rpn_boxes, rpn_scores)
        
            rois = box_utils.to_absolute_coordinates(rois, height, width)

            return rois, rois_scores

    def get_boxes(self, outputs, image_info):
        with tf.name_scope("get_boxes"):
            predicted_boxes = tf.cast(outputs["rcnn_boxes"], tf.float32)
            predicted_labels = tf.cast(outputs["rcnn_labels"], tf.float32)
            rois = tf.cast(outputs["rois"], tf.float32)
        
            input_size = image_info["input_size"]
            valid_height = image_info["valid_size"][:, 0:1, None]
            valid_width = image_info["valid_size"][:, 1:2, None]
            predicted_boxes = self.rcnn_head.bbox_head.bbox_decoder(rois, predicted_boxes)
            rpn_boxes = box_utils.clip_boxes(rpn_boxes, valid_height, valid_width)
            predicted_boxes = box_utils.to_normalized_coordinates(
                predicted_boxes, input_size[:, 0:1, None], input_size[:, 1:2, None])
            predicted_boxes = tf.clip_by_value(predicted_boxes, 0, 1)

            if self.rcnn_head.bbox_head.use_sigmoid:
                predicted_scores = tf.nn.sigmoid(predicted_labels)
            else:
                predicted_scores = tf.nn.softmax(predicted_labels, axis=-1)
                predicted_scores = predicted_scores[:, :, 1:]
            
            return self.nms(predicted_boxes, predicted_scores)

 