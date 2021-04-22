import tensorflow as tf 
from utils import box_utils
from ..head import BaseHead
from core import build_nms
from core import build_anchor_generator 
 

class AnchorHead(BaseHead):
    def __init__(self, cfg, test_cfg=None, anchor_cfg=None, num_classes=80, **kwargs):
        super(AnchorHead, self).__init__(cfg=cfg, 
                                         test_cfg=test_cfg, 
                                         anchor_cfg=anchor_cfg, 
                                         num_classes=num_classes, **kwargs)

        self.anchor_cfg = anchor_cfg

    def _init_anchor_generators(self):
        strides = self.anchor_cfg.strides
        scales = self.anchor_cfg.get("scales")
        anchor_kwargs = self.anchor_cfg.as_dict()
        anchor_kwargs.pop("strides")
        if scales:
            anchor_kwargs.pop("scales")
        self.anchor_generators = [
            build_anchor_generator(name="anchor_generator%d" % level, 
                                   strides=strides[i],
                                   scales=scales[i] if scales else None, 
                                   **anchor_kwargs) 
            for i, level in enumerate(range(self.min_level, self.max_level + 1))
        ]

    def _make_init_layers(self):
        self.classifier = tf.keras.layers.Conv2D(
            filters=self.num_anchors * self._label_dims,
            kernel_size=(1, 1),
            strides=(1, 1),
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
            name="predicted_class")

        self.regressor = tf.keras.layers.Conv2D(
            filters=self.num_anchors * 4,
            kernel_size=(1, 1),
            strides=(1, 1),
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
            name="predicted_box")

    @property
    def num_anchors(self):
        return self.anchor_cfg.num_anchors
        
    def get_targets(self, gt_boxes, gt_labels, total_anchors):
        with tf.name_scope("gt_targets"):
            batch_size = tf.shape(gt_boxes)[0]
            target_boxes_ta = tf.TensorArray(tf.float32, batch_size)
            target_labels_ta = tf.TensorArray(tf.int64, batch_size)
            label_weights_ta = tf.TensorArray(tf.float32, batch_size)
            box_weights_ta = tf.TensorArray(tf.float32, batch_size)

            for i in tf.range(batch_size):
                t_boxes, t_labels = self.assigner(gt_boxes[i], gt_labels[i], total_anchors)
                t_boxes, t_labels, b_weights, l_weights = self.sampler(t_boxes, t_labels)
            
                target_boxes_ta = target_boxes_ta.write(i, t_boxes)
                target_labels_ta = target_labels_ta.write(i, t_labels)
                box_weights_ta = box_weights_ta.write(i, b_weights)
                label_weights_ta = label_weights_ta.write(i, l_weights)
            
            target_boxes = target_boxes_ta.stack()
            target_labels = target_labels_ta.stack()
            box_weights = box_weights_ta.stack()
            label_weights = label_weights_ta.stack()

            target_boxes = tf.stop_gradient(target_boxes)
            target_labels = tf.stop_gradient(target_labels)
            box_weights = tf.stop_gradient(box_weights)
            label_weights = tf.stop_gradient(label_weights)

            return target_boxes, target_labels, box_weights, label_weights

    def compute_losses(self, predictions, image_info):
        with tf.name_scope("compute_losses"):
            predicted_boxes = tf.cast(predictions["boxes"], tf.float32)
            predicted_labels = tf.cast(predictions["labels"], tf.float32)
            total_anchors = tf.cast(predictions["anchors"], tf.float32)

            gt_boxes = image_info["boxes"]
            gt_labels = image_info["labels"]

            target_boxes, target_labels, box_weights, label_weights = self.get_targets(
                gt_boxes, gt_labels, total_anchors)
            
            if self.use_sigmoid:
                target_labels = tf.one_hot(target_labels - 1, self.num_classes)
            else:
                target_labels = tf.one_hot(target_labels, self.num_classes + 1) 
            
            predicted_boxes = tf.concat([predicted_boxes[..., 1:2],
                                         predicted_boxes[..., 0:1],
                                         predicted_boxes[..., 3:4],
                                         predicted_boxes[..., 2:3]], -1)
            if self._use_iou_loss:
                predicted_boxes = self.bbox_decoder(total_anchors[None], predicted_boxes)
            else:
                target_boxes = self.bbox_encoder(total_anchors[None], target_boxes)

            label_loss = self.label_loss_func(target_labels, predicted_labels, label_weights)
            bbox_loss = self.bbox_loss_func(target_boxes, predicted_boxes, box_weights)
           
            label_loss = tf.reduce_sum(label_loss) / (tf.reduce_sum(box_weights) + 1.)
            bbox_loss = tf.reduce_sum(bbox_loss)  / (tf.reduce_sum(box_weights) + 1.)
            
            return dict(bbox_loss=bbox_loss, label_loss=label_loss)

    def get_boxes(self, outputs):
        with  tf.name_scope("get_boxes"):
            predicted_boxes_list = []
            predicted_labels_list = []
            for _, level in enumerate(range(self.min_level, self.max_level+1)):
                pred_boxes = tf.cast(outputs["boxes"]["level%d" % level], tf.float32)
                pred_labels = tf.cast(outputs["labels"]["level%d" % level], tf.float32)
                anchors = tf.cast(outputs["total_anchors"]["level%d" % level], tf.float32)

                h, w = tf.shape(pred_boxes)[1], tf.shape(pred_boxes)[2]

                pred_boxes = tf.reshape(pred_boxes, [-1, h * w * self.num_anchors, 4])
                pred_labels = tf.reshape(pred_labels, [-1, h * w * self.num_anchors, self._label_dims])

                pred_boxes = tf.concat([pred_boxes[..., 1:2],
                                        pred_boxes[..., 0:1],
                                        pred_boxes[..., 3:4],
                                        pred_boxes[..., 2:3]], -1)
                pred_boxes = self.bbox_decoder(anchors[None], pred_boxes)
                
                # input_size = tf.convert_to_tensor([[h, w]], pred_boxes.dtype) * (2 ** level)
                # pred_boxes = box_utils.to_normalized_coordinates(
                #     pred_boxes, input_size[:, 0:1, None], input_size[:, 1:2, None])
                # pred_boxes = tf.clip_by_value(pred_boxes, 0, 1)
            
                predicted_boxes_list.append(pred_boxes)
                predicted_labels_list.append(pred_labels)
            
            predicted_boxes = tf.concat(predicted_boxes_list, 1)
            predicted_labels = tf.concat(predicted_labels_list, 1)
            
            if self.use_sigmoid:
                predicted_scores = tf.nn.sigmoid(predicted_labels)
            else:
                predicted_scores = tf.nn.softmax(predicted_labels, axis=-1)
                predicted_scores = predicted_scores[:, :, 1:]  
            
            return self.nms(predicted_boxes, predicted_scores)
 
