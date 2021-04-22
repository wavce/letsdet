import math
import tensorflow as tf
from ...builder import HEADS
from ..head import BaseHead 
from core import build_loss
from core.layers import Scale
from ...common import ConvNormActBlock


@HEADS.register
class CenterHeatmapHead(BaseHead):
    def __init__(self, cfg, num_classes, **kwargs):
        super(CenterHeatmapHead, self).__init__(cfg=cfg, num_classes=num_classes, **kwargs)

        self.hm_loss_func = build_loss(**cfg.hm_loss.as_dict())
        self.wh_loss_func = build_loss(**cfg.wh_loss.as_dict())
        self.reg_loss_func = build_loss(**cfg.reg_loss.as_dict())
        
        bias_value = -2.19  # -math.log((1 - 0.1) / 0.1)
        
        if cfg.num_stacks > 1:
            for i in range(cfg.num_stacks):
                hm = tf.keras.Sequential(name="hm/%d" % i) 
                wh = tf.keras.Sequential(name="wh/%d" % i)
                reg = tf.keras.Sequential(name="reg/%d" % i)
               
                for j in range(cfg.repeats):
                    hm.add(ConvNormActBlock(filters=cfg.feat_dims,
                                            kernel_size=3,
                                            data_format=self.data_format,
                                            normalization=cfg.normalization.as_dict() if cfg.normalization else None,
                                            activation=cfg.activation.as_dict(),
                                            name="%d" % j))
                    wh.add(ConvNormActBlock(filters=cfg.feat_dims,
                                            kernel_size=3,
                                            data_format=self.data_format,
                                            normalization=cfg.normalization.as_dict() if cfg.normalization else None,
                                            activation=cfg.activation.as_dict(),
                                            name="%d" % j))
                    reg.add(ConvNormActBlock(filters=cfg.feat_dims,
                                            kernel_size=3,
                                            data_format=self.data_format,
                                            normalization=cfg.normalization.as_dict() if cfg.normalization else None,
                                            activation=cfg.activation.as_dict(),
                                            name="%d" % j))
               
                hm.add(tf.keras.layers.Conv2D(filters=num_classes,
                                              kernel_size=1,
                                              data_format=self.data_format,
                                              bias_initializer=tf.keras.initializers.Constant(bias_value),
                                              name="predicted_hm%d" % cfg.repeats))
                wh.add(tf.keras.layers.Conv2D(filters=2,
                                              kernel_size=1,
                                              data_format=self.data_format,
                                              name="predicted_wh%d" % cfg.repeats))
                reg.add(tf.keras.layers.Conv2D(filters=2,
                                               kernel_size=1,
                                               data_format=self.data_format,
                                               name="predicted_reg%d" % cfg.repeats))
               
                setattr(self, "wh%d" % i, wh)
                setattr(self, "hm%d" % i, hm)
                setattr(self, "reg%d" % i, reg)
        else:            
            hm = tf.keras.Sequential(name="hm") 
            wh = tf.keras.Sequential(name="wh")
            reg = tf.keras.Sequential(name="reg")
           
            for j in range(cfg.repeats):
                hm.add(ConvNormActBlock(filters=cfg.feat_dims,
                                        kernel_size=3,
                                        data_format=self.data_format,
                                        normalization=cfg.normalization.as_dict() if cfg.normalization else None,
                                        activation=cfg.activation.as_dict(),
                                        name="%d" % j))
                wh.add(ConvNormActBlock(filters=cfg.feat_dims,
                                        kernel_size=3,
                                        data_format=self.data_format,
                                        normalization=cfg.normalization.as_dict() if cfg.normalization else None,
                                        activation=cfg.activation.as_dict(),
                                        name="%d" % j))
                reg.add(ConvNormActBlock(filters=cfg.feat_dims,
                                        kernel_size=3,
                                        data_format=self.data_format,
                                        normalization=cfg.normalization.as_dict() if cfg.normalization else None,
                                        activation=cfg.activation.as_dict(),
                                        name="%d" % j))
            hm.add(tf.keras.layers.Conv2D(filters=num_classes,
                                          kernel_size=1,
                                          data_format=self.data_format,
                                          bias_initializer=tf.keras.initializers.Constant(bias_value),
                                          name="predicted_hm"))
            wh.add(tf.keras.layers.Conv2D(filters=2,
                                          kernel_size=1,
                                          data_format=self.data_format,
                                          name="predicted_wh"))
            reg.add(tf.keras.layers.Conv2D(filters=2,
                                           kernel_size=1,
                                           data_format=self.data_format,
                                           name="predicted_reg"))
            setattr(self, "wh", wh)
            setattr(self, "hm", hm)
            setattr(self, "reg", reg)

    def call(self, inputs, training=None):
        outputs = {"hm": [], "wh": [], "reg": []}
        if self.cfg.num_stacks > 1:
            for i in range(self.cfg.num_stacks):            
                hm = getattr(self, "hm%d" % i)(inputs[i], training=training)
                wh = getattr(self, "wh%d" % i)(inputs[i], training=training)
                reg = getattr(self, "reg%d" % i)(inputs[i], training=training)
                
                outputs["hm"].append(hm)
                outputs["wh"].append(wh)
                outputs["reg"].append(reg)
        else:
            hm = getattr(self, "hm")(inputs, training=training)
            wh = getattr(self, "wh")(inputs, training=training)
            reg = getattr(self, "reg")(inputs, training=training)
        
            outputs["hm"].append(hm)
            outputs["wh"].append(wh)
            outputs["reg"].append(reg)

        if self.is_training:
            return outputs
        
        return self.get_boxes(outputs)
    
    def get_targets(self, feat_height, feat_width, gt_boxes, gt_labels):
        with tf.name_scope("gt_targets"):
             batch_size = tf.shape(gt_boxes)[0]
             target_hm_ta = tf.TensorArray(tf.float32, batch_size)
             target_wh_ta = tf.TensorArray(tf.float32, batch_size)
             target_reg_ta = tf.TensorArray(tf.float32, batch_size)
            #  pos_inds_ta = tf.TensorArray(tf.int32, batch_size)
            #  reg_mask_ta = tf.TensorArray(tf.int32, batch_size)

             for i in tf.range(batch_size):
                  t_hm, t_wh, t_reg = self.assigner(feat_height, feat_width, gt_boxes[i], gt_labels[i])
            
                  target_hm_ta = target_hm_ta.write(i, t_hm)
                  target_wh_ta = target_wh_ta.write(i, t_wh)
                  target_reg_ta = target_reg_ta.write(i, t_reg)
                #   pos_inds_ta = pos_inds_ta.write(i, inds)
                #   reg_mask_ta = reg_mask_ta.write(i, mask)
                
             target_hm = target_hm_ta.stack()
             target_wh = target_wh_ta.stack()
             target_reg = target_reg_ta.stack()
            #  pos_inds = pos_inds_ta.stack()
            #  reg_mask = reg_mask_ta.stack()
             
             target_hm = tf.stop_gradient(target_hm)
             target_wh = tf.stop_gradient(target_wh)
             target_reg = tf.stop_gradient(target_reg)

             return target_hm, target_wh, target_reg  # , pos_inds, reg_mask

    def compute_losses(self, predictions, image_info):
        with tf.name_scope("compute_losses"):
            gt_boxes = image_info["boxes"]
            gt_labels = image_info["labels"]

            feat_h = tf.shape(predictions["hm"][0])[1:3]
            target_hm, target_wh, target_reg = self.get_targets(feat_h[0], feat_h[1], gt_boxes, gt_labels - 1)

            fpos_mask = tf.cast(tf.reduce_max(target_hm, -1, True) == 1, tf.float32)
            num_pos = tf.reduce_sum(fpos_mask) + 1.
        
            hm_loss_list = []
            wh_loss_list = []
            reg_loss_list = []
            for i in range(self.cfg.num_stacks):
                predicted_hm = tf.cast(predictions["hm"][i], tf.float32)
                predicted_wh = tf.cast(predictions["wh"][i], tf.float32)
                predicted_reg = tf.cast(predictions["reg"][i], tf.float32)
                                
                hm_loss = self.hm_loss_func(target_hm, predicted_hm)
                hm_loss = tf.reduce_sum(hm_loss) / num_pos
            
                wh_loss = self.wh_loss_func(target_wh, predicted_wh) * fpos_mask
                reg_loss = self.reg_loss_func(target_reg, predicted_reg) * fpos_mask

                wh_loss = tf.reduce_sum(wh_loss) / num_pos 
                reg_loss = tf.reduce_sum(reg_loss) / num_pos
                
                hm_loss_list.append(hm_loss)
                wh_loss_list.append(wh_loss)
                reg_loss_list.append(reg_loss)
            
            hm_loss = tf.add_n(hm_loss_list) / self.cfg.num_stacks
            wh_loss = tf.add_n(wh_loss_list) / self.cfg.num_stacks
            reg_loss = tf.add_n(reg_loss_list) / self.cfg.num_stacks
            
            return dict(hm_loss=hm_loss, wh_loss=wh_loss, reg_loss=reg_loss)

    def get_boxes(self, outputs):
        with tf.name_scope("get_boxes"):
            predicted_hm = tf.cast(outputs["hm"][-1], tf.float32)
            predicted_wh = tf.cast(outputs["wh"][-1], tf.float32)
            predicted_reg = tf.cast(outputs["reg"][-1], tf.float32)
            data_format = "NHWC" if self.data_format == "channels_last" else "NCHW"
            predicted_hm = tf.nn.sigmoid(predicted_hm)
            pool_hm = tf.nn.max_pool2d(input=predicted_hm, 
                                       ksize=(1, 3, 3, 1), 
                                       strides=(1, 1, 1, 1), 
                                       padding="SAME", 
                                       data_format=data_format)
            keep = tf.cast(tf.equal(pool_hm, predicted_hm), predicted_hm.dtype)
            predicted_hm *= keep

            topk = self.test_cfg.topk
            shape = tf.shape(predicted_hm)
            b, h, w, ncls = shape[0], shape[1], shape[2], shape[3]
            
            predicted_hm = tf.transpose(predicted_hm, [0, 3, 1, 2])
            topk_scores, topk_inds = tf.nn.top_k(
               tf.reshape(predicted_hm, [b, ncls, h * w]), k=topk)
            topk_inds = topk_inds % (h * w)
            topk_ys = tf.cast(topk_inds // w, tf.float32)
            topk_xs = tf.cast(topk_inds % w, tf.float32)
         
            topk_score, topk_ind = tf.nn.top_k(
               tf.reshape(topk_scores, [b, -1]), k=topk)
            topk_classes = topk_ind // topk
            batch_inds = tf.reshape(tf.range(b), [b, 1])
            topk_ind = tf.reshape(batch_inds * topk * ncls + topk_ind, [-1])
            topk_inds = tf.gather(tf.reshape(topk_inds + batch_inds[:, :, None] * w * h, [-1]), topk_ind)

            topk_ys = tf.gather(tf.reshape(topk_ys, [-1]), topk_ind)
            topk_xs = tf.gather(tf.reshape(topk_xs, [-1]), topk_ind)

            wh = tf.gather(tf.reshape(predicted_wh, [b * h * w, 2]), topk_inds)
            reg = tf.gather(tf.reshape(predicted_reg, [b * h * w, 2]), topk_inds)
            ys = topk_ys + reg[..., 1]
            xs = topk_xs + reg[..., 0]
            
            w = tf.cast(w, xs.dtype)
            h = tf.cast(h, ys.dtype)
            topk_boxes = tf.stack([
                tf.clip_by_value(xs - wh[..., 0] * 0.5, 0, w),
                tf.clip_by_value(ys - wh[..., 1] * 0.5, 0, h),
                tf.clip_by_value(xs + wh[..., 0] * 0.5, 0, w),
                tf.clip_by_value(ys + wh[..., 1] * 0.5, 0, h)], axis=-1)
            
            topk_boxes = tf.reshape(topk_boxes, [b, topk, 4]) * self.cfg.downsample_ratio
            # topk_boxes = tf.clip_by_value(topk_boxes, 0, 1)

            thresh_mask = tf.cast(topk_score > self.test_cfg.score_threshold, tf.float32)
            valid_detections = tf.cast(tf.reduce_sum(thresh_mask, -1), tf.int32)
            topk_boxes = tf.expand_dims(thresh_mask, -1) * topk_boxes
            topk_score = thresh_mask * topk_score
            topk_classes = tf.cast(thresh_mask, topk_classes.dtype) * topk_classes

            return dict(nmsed_boxes=topk_boxes,
                        nmsed_scores=topk_score, 
                        nmsed_classes=topk_classes, 
                        valid_detections=valid_detections)



            
         