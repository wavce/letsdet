from abc import ABCMeta
from abc import abstractmethod
import tensorflow as tf

from .anchor_head import AnchorHead
from utils import box_utils
from ...builder import HEADS
from core.anchors import AnchorGenerator


@HEADS.register
class RPNHead(AnchorHead):
    def __init__(self, cfg, anchor_cfg, **kwargs):
        super(RPNHead, self).__init__(num_classes=cfg.num_classes, 
                                      cfg=cfg, 
                                      anchor_cfg=anchor_cfg, 
                                      **kwargs)
        self._init_anchor_generators()

        self.conv1 = tf.keras.layers.Conv2D(
            filters=self.cfg.feat_dims, 
            kernel_size=(3, 3), 
            strides=(1, 1), 
            padding="same",
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
            activation=cfg.activation.activation,
            name="rpn_conv")
        
        self._make_init_layers()
    
    def call(self, inputs, training=None):
        predicted_boxes = dict()
        predicted_labels = dict()
        total_anchors = dict()
        
        for i, level in enumerate(range(self.min_level, self.max_level + 1)):
            x = self.conv1(inputs[i])
            p_boxes = self.regressor(x)
            p_scores = self.classifier(x)
            
            if self.data_format == "channels_first":
                p_boxes = tf.transpose(p_boxes, [0, 2, 3, 1])
                p_scores = tf.transpose(p_scores, [0, 2, 3, 1])

            feat_h, feat_w = tf.shape(p_boxes)[1], tf.shape(p_boxes)[2]
            anchors = self.anchor_generators[i](feat_h, feat_w)
            
            predicted_boxes["level%d" % level] = p_boxes
            predicted_labels["level%d" % level] = p_scores
            total_anchors["level%d" % level] = anchors

        outputs = dict(boxes=predicted_boxes, labels=predicted_labels, total_anchors=total_anchors)
        
        return outputs, self.get_proposals(outputs)
        
    def compute_losses(self, outputs, image_info):
        with tf.name_scope("compute_losses"):
            predicted_boxes = tf.cast(outputs["boxes"], tf.float32)
            predicted_labels = tf.cast(outputs["labels"], tf.float32)
            total_anchors = tf.cast(outputs["total_anchors"], tf.float32)

            gt_boxes = tf.cast(image_info["boxes"], tf.float32)
            gt_labels = tf.cast(image_info["labels"], tf.int64)

            target_boxes, target_labels, box_weights, label_weights = self.get_targets(
                gt_boxes, gt_labels, total_anchors)
            
            target_labels = tf.cast(target_labels > 0, tf.int32)
            if self.use_sigmoid:
                target_labels = tf.cast(tf.expand_dims(target_labels, -1), tf.float32)
            else:
                target_labels = tf.one_hot(target_labels, 2)
            
            if self._use_iou_loss:
                predicted_boxes = self.bbox_decoder(total_anchors[None], predicted_boxes)
            else:
                target_boxes = self.bbox_encoder(total_anchors[None], target_boxes)

            label_loss = self.label_loss_func(target_labels, predicted_labels, label_weights)
            bbox_loss = self.bbox_loss_func(target_boxes, predicted_boxes, box_weights) 

            label_loss = tf.reduce_sum(label_loss) / (tf.reduce_sum(box_weights) + 1.)
            bbox_loss = tf.reduce_sum(bbox_loss)  / (tf.reduce_sum(box_weights) + 1.)
            
            outputs = dict(rpn_bbox_loss=bbox_loss, rpn_label_loss=label_loss)

            return outputs, self.get_proposals(outputs)

    def get_proposals(self, outputs):
        with tf.name_scope("get_proposals"):
            topk_boxes_list = []
            topk_scores_list = []
            pre_nms_size = (
                self.cfg.train_proposal.pre_nms_size 
                if self.is_training 
                else self.cfg.test_proposal.pre_nms_size)
            post_nms_size = (
                self.cfg.train_proposal.post_nms_size 
                if self.is_training 
                else self.cfg.test_proposal.post_nms_size)
            iou_threshold = (
                self.cfg.train_proposal.iou_threshold 
                if self.is_training 
                else self.cfg.test_proposal.iou_threshold)
            
            for level in range(self.min_level, self.max_level + 1):
                p_boxes = outputs["boxes"]["level%d" % level]
                p_scores = outputs["labels"]["level%d" % level]
                anchors = outputs["total_anchors"]["level%d" % level]
                p_boxes = tf.cast(p_boxes, tf.float32)
                p_scores = tf.cast(p_scores, tf.float32)
                anchors = tf.cast(anchors, tf.float32)

                feat_h, feat_w = tf.shape(p_boxes)[1], tf.shape(p_boxes)[2]
                hi_wi_a = feat_h * feat_w * self.num_anchors
                p_boxes = tf.reshape(p_boxes, [-1, hi_wi_a, 4])
                p_boxes = tf.concat([p_boxes[..., 1:2], 
                                     p_boxes[..., 0:2], 
                                     p_boxes[..., 3:4], 
                                     p_boxes[..., 2:3]], -1)
              
                p_boxes = self.bbox_decoder(anchors[None], p_boxes)

                img_size = tf.convert_to_tensor([[[feat_h, feat_w] * 2]], p_boxes.dtype) * (2 ** level)
                p_boxes = tf.minimum(tf.maximum(p_boxes, 0), img_size)
                p_scores = tf.reshape(p_scores, [-1, hi_wi_a, self._label_dims])
                # if self.use_sigmoid:
                #     p_scores = tf.nn.sigmoid(p_scores)
                # else:
                #     p_scores = tf.nn.softmax(p_scores, axis=-1)
                #     p_scores = p_scores[:, :, 1:]  

                p_scores = tf.squeeze(p_scores, -1)
                pre_nms_size = tf.minimum(pre_nms_size, hi_wi_a)
                topk_scores, topk_inds = tf.nn.top_k(p_scores, pre_nms_size)
                
                bs = tf.shape(p_scores)[0]
                batch_inds = tf.tile(tf.reshape(tf.range(bs), [bs, 1]), [1, pre_nms_size])
                inds = tf.stack([batch_inds, topk_inds], -1)
                topk_boxes = tf.gather_nd(p_boxes, inds)

                topk_boxes_list.append(topk_boxes)
                topk_scores_list.append(topk_scores)

            topk_boxes = tf.concat(topk_boxes_list, 1)
            topk_scores = tf.concat(topk_scores_list, 1)
                  
            # def _single_image_nms(boxes, scores):
            #     selected_inds = tf.image.non_max_suppression(boxes, scores, post_nms_size, iou_threshold, 0.)
            #     return tf.gather(boxes, selected_inds), tf.gather(scores, selected_inds)
            
            # proposals, scores = tf.map_fn(lambda inp: _single_image_nms(inp[0], inp[1]), 
            #                               elems=(topk_boxes, topk_scores), 
            #                               fn_output_signature=(tf.float32, tf.float32))

            proposals, scores, _, _ = tf.image.combined_non_max_suppression(
                boxes=tf.expand_dims(topk_boxes, 2),
                scores=tf.expand_dims(topk_scores, -1),
                max_output_size_per_class=post_nms_size,
                max_total_size=post_nms_size,
                iou_threshold=iou_threshold,
                clip_boxes=False)
            tf.print(proposals)

            return dict(rois=proposals, scores=scores)

