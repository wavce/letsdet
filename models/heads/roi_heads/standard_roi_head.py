import tensorflow as tf 
from ...builder import HEADS
from utils import box_utils
from ...common import ConvNormActBlock
from core.layers import build_roi_pooling
from ..roi_heads.base_roi_head import BaseRoIHead


@HEADS.register
class StandardRoIHead(BaseRoIHead):
    """Simplest base roi head containing one bbox head and one mask head."""
    def __init__(self, cfg, **kwargs):
        super(StandardRoIHead, self).__init__(cfg, **kwargs)

        self._make_init_layers()
    
    def _make_init_layers(self):
        self.classifier = tf.keras.layers.Dense(units=self._label_dims, name="classifier")

        num_boxes = 1 if self.cfg.class_agnostic else self.num_classes
        self.regressor = tf.keras.layers.Dense(units=4 * num_boxes, name="regressor")

    def _make_bbox_head(self, bbox_head_cfg):
        self.bbox_roi_pooling = build_roi_pooling(min_level=self.min_level, 
                                                  max_level=self.max_level,
                                                  **bbox_head_cfg.roi_pooling.as_dict())
        self.bbox_head = tf.keras.Sequential(name="box_head")

        for i in range(bbox_head_cfg.num_convs):
            self.bbox_head.add(ConvNormActBlock(filters=bbox_head_cfg.conv_dims,
                                                kernel_size=3,
                                                normalization=bbox_head_cfg.normalization.as_dict() if bbox_head_cfg.normalization else None,
                                                activation=bbox_head_cfg.activation.as_dict() if bbox_head_cfg.activation else None,
                                                dropblock=bbox_head_cfg.dropblock,
                                                name="conv" + str(i + 1)))
        for i in range(bbox_head_cfg.num_fc):
            if i == 0:
                self.bbox_head.add(tf.keras.layers.Flatten(data_format=self.data_format))
            self.bbox_head.add(tf.keras.layers.Dense(units=bbox_head_cfg.fc_dims,
                                                     activation=bbox_head_cfg.activation.activation if bbox_head_cfg.activation else None,
                                                     name="fc" + str(i + 1)))

    def _make_mask_head(self, mask_head_cfg):
        raise NotImplementedError()      
    
    def call(self, inputs, training=None):        
        features_list, proposals = inputs

        features_list = [features_list[i] for i in range(self.max_level - self.min_level + 1)]

        rois = proposals["rois"]
        bs = tf.shape(rois)[0]
        outputs = dict(rois=rois)
        if self.has_bbox_head:
            box_feats = self.bbox_roi_pooling(features_list, rois) 
            filters = self.cfg.bbox_head.roi_pooling.feat_dims
            num_boxes = tf.shape(box_feats)[1]
            box_feats = tf.reshape(box_feats, [-1, self.pooled_size, self.pooled_size, filters])
            box_feats = tf.transpose(box_feats, [0, 3, 1, 2])
            
            feats = self.bbox_head(box_feats, training=training)
            predicted_boxes = self.regressor(feats)
            predicted_labels = self.classifier(feats)

            predicted_boxes = tf.reshape(predicted_boxes, [bs, num_boxes, self.num_classes, 4])
            predicted_labels = tf.reshape(predicted_labels, [bs, num_boxes, self._label_dims])

            outputs["boxes"] = predicted_boxes
            outputs["labels"] = predicted_labels            
        
        # if self.has_mask_head:
        #     if not self.share_roi_pooling:
        #         pos_roi = None
        #         mask_feats = self.mask_roi_pooling(features_list, pos_roi)

        if self.is_training:
            return outputs
        
        return self.get_boxes(outputs)
    
    def get_targets(self, gt_boxes, gt_labels, rois):
        raise NotImplementedError()

    def compute_losses(self, outputs, image_info):
        raise NotImplementedError()
    
    def get_boxes(self, outputs):
        with tf.name_scope("get_boxes"):
            pred_boxes = tf.cast(outputs["boxes"], tf.float32)
            pred_labels = tf.cast(outputs["labels"], tf.float32)
            rois = tf.cast(outputs["rois"], tf.float32)

            pred_boxes = tf.concat([pred_boxes[..., 1:2],
                                    pred_boxes[..., 0:1],
                                    pred_boxes[..., 3:4],
                                    pred_boxes[..., 2:3]], -1)
            if not self.cfg.class_agnostic:
                rois = tf.expand_dims(rois, 2)
            
            pred_boxes = self.bbox_decoder(rois, pred_boxes)
            
            # input_size = tf.convert_to_tensor([[h, w]], pred_boxes.dtype) * (2 ** level)
            # pred_boxes = box_utils.to_normalized_coordinates(
            #     pred_boxes, input_size[:, 0:1, None], input_size[:, 1:2, None])
            # pred_boxes = tf.clip_by_value(pred_boxes, 0, 1)
        
            if self.use_sigmoid:
                predicted_scores = tf.nn.sigmoid(pred_labels)
            else:
                predicted_scores = tf.nn.softmax(pred_labels, axis=-1)
                predicted_scores = predicted_scores[:, :, 1:]  
                        
            return self.nms(pred_boxes, predicted_scores)

    