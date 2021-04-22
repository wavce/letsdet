import time
import math
import numpy as np
import tensorflow as tf
from core import metrics
from data import build_dataset
from core import build_optimizer
from models import build_detector
from core import build_learning_rate_scheduler


class SingleGPUTrainer(object):
    """Train class.

        Args:
            cfg: the configuration cfg.
        """

    def __init__(self, cfg, logger):
        self.logger = logger
        use_mixed_precision = cfg.dtype in ["float16", "FP16"]
        if use_mixed_precision:
            tf.keras.mixed_precision.set_global_policy("mixed_float16") 
            print("Using mixed precision training.")

        self.train_dataset = build_dataset(dtype=tf.float16 if use_mixed_precision else tf.float32,
                                           **cfg.train.dataset.as_dict())
        self.val_dataset = build_dataset(dtype=tf.float16 if use_mixed_precision else tf.float32,
                                         **cfg.val.dataset.as_dict())
       
        if cfg.train.get("proposal_layer"):
            self.detector = build_detector(
                cfg.detector, cfg=cfg, proposal_cfg=cfg.train.proposal_layer)
        else:
            self.detector = build_detector(cfg.detector, cfg=cfg)

        self.detector.load_pretrained_weights(cfg.train.pretrained_weights_path)

        train_steps_per_epoch = cfg.train.dataset.num_samples // cfg.train.dataset.batch_size
        self.total_train_steps = cfg.train.scheduler.train_epochs * train_steps_per_epoch
        self.warmup_steps = cfg.train.scheduler.warmup.steps
        self.warmup_learning_rate = cfg.train.scheduler.warmup.warmup_learning_rate
        self.learning_rate_scheduler = build_learning_rate_scheduler(
            **cfg.train.scheduler.learning_rate_scheduler.as_dict(), 
            train_steps=self.total_train_steps, 
            warmup_steps=self.warmup_steps,
            train_steps_per_epoch=train_steps_per_epoch)

        optimizer = build_optimizer(learning_rate=self.learning_rate_scheduler, **cfg.train.optimizer.as_dict())

        if use_mixed_precision:
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer, dynamic=True) 
            self.logger.info("Using mixed precision training.")

        self.optimizer = optimizer
        self.use_mixed_precision = use_mixed_precision
        self.cfg = cfg

        self.global_step = tf.Variable(
            initial_value=0, trainable=False, name="global_step", dtype=tf.int64)

        self.val_steps = tf.Variable(
            0, trainable=False, name="val_steps", dtype=tf.int64)
       
        self.checkpoint = tf.train.Checkpoint(
            optimizer=self.optimizer, 
            detector=self.detector.detector,
            global_step=self.global_step,
            val_steps=self.val_steps)
        self.manager = tf.train.CheckpointManager(
            checkpoint=self.checkpoint, 
            directory=cfg.train.checkpoint_dir, 
            max_to_keep=10)

        self.epochs = 0
        latest_checkpoint = self.manager.latest_checkpoint
        if latest_checkpoint is not None:
            # try:
            #     steps = int(latest_checkpoint.split("-")[-1])
            #     self.global_step.assign(steps)
            #     self.epochs = steps // train_steps_per_epoch
            # except:
            #     self.global_step.assign(0)
            self.checkpoint.restore(latest_checkpoint)
            self.logger.info("Restored weights from %s.", latest_checkpoint)
        else:
            self.global_step.assign(0)

        self.summary_writer = tf.summary.create_file_writer(logdir=cfg.train.summary_dir)
        self.log_every_n_steps = cfg.train.log_every_n_steps
        self.use_jit = tf.config.optimizer.get_jit() is not None

        self.train_loss_metrics = {"l2_loss": tf.keras.metrics.Mean(), "loss": tf.keras.metrics.Mean()}
        self.val_loss_metrics = {"l2_loss": tf.keras.metrics.Mean(), "loss": tf.keras.metrics.Mean()}
        self.ap_metric = None 
        self._add_graph = True
        self.ap_metric = metrics.mAP(self.cfg.num_classes)
    
    def _exclude_weights(self, name, excluding_weight_names=[]):
        for n in excluding_weight_names:
            if n in name:
                return False
        
        return True
    
    @tf.function(experimental_relax_shapes=True)
    def train_step(self, images, image_info):
        with tf.GradientTape() as tape:
            outputs = self.detector.detector(images, training=True)
            
            losses = self.detector.compute_losses(outputs, image_info)
            l2_loss = tf.add_n([
                tf.nn.l2_loss(w) for w in self.detector.detector.trainable_weights 
                if self._exclude_weights(w.name, self.cfg.excluding_weight_names) and "kernel" in w.name
            ]) * self.cfg.weight_decay

            loss = tf.add_n([v for k, v in losses.items() if "loss" in k] + [l2_loss])
            
            if self.use_mixed_precision:
                scaled_loss = self.optimizer.get_scaled_loss(loss)
            else:
                scaled_loss = loss

        gradients = tape.gradient(scaled_loss, self.detector.detector.trainable_weights)

        if self.use_mixed_precision:
            gradients = self.optimizer.get_unscaled_gradients(gradients)
        
        if self.cfg.train.gradient_clip_norm > 0.0:
            gradients, _ = tf.clip_by_global_norm(gradients, self.cfg.train.gradient_clip_norm)

        self.optimizer.apply_gradients(zip(gradients, self.detector.detector.trainable_variables))
        
        for key, value in losses.items():
            if "loss" not in key: continue
            if key not in self.train_loss_metrics:
                self.train_loss_metrics[key] = tf.keras.metrics.Mean()
            self.train_loss_metrics[key].update_state(value)
        self.train_loss_metrics["loss"].update_state(loss)
        self.train_loss_metrics["l2_loss"].update_state(l2_loss)

        if tf.equal(self.global_step.value() % self.log_every_n_steps, 0):
            x = self.detector.head.get_boxes(outputs)
                    
            normalized_factor = tf.convert_to_tensor(
                [tf.shape(images)[1], tf.shape(images)[2]] * 2, tf.float32)
            gt_boxes = image_info["boxes"] 
            pred_boxes = x["nmsed_boxes"]
            gt_boxes = tf.concat([
                gt_boxes[..., 1:2], 
                gt_boxes[..., 0:1], 
                gt_boxes[..., 3:4], 
                gt_boxes[..., 2:3]], 
                -1) / normalized_factor
            pred_boxes = tf.concat([
                pred_boxes[..., 1:2], 
                pred_boxes[..., 0:1], 
                pred_boxes[..., 3:4], 
                pred_boxes[..., 2:3]], 
                -1) / normalized_factor

            images = tf.image.draw_bounding_boxes(images, gt_boxes, tf.constant([[255., 0., 0.]]))
            images = tf.image.draw_bounding_boxes(images, pred_boxes, tf.constant([[0., 255., 0.]]))    
            images = tf.cast(images, tf.uint8)

            with tf.device("/cpu:0"):
                with self.summary_writer.as_default():
                    tf.summary.image("train/images", images, self.global_step, 5)

        return loss

    @tf.function(experimental_relax_shapes=True)
    def val_step(self, images, image_info):
        outputs = self.detector.detector(images, training=False)
        losses = self.detector.compute_losses(outputs, image_info)

        l2_loss = tf.add_n([
            tf.nn.l2_loss(w) for w in self.detector.detector.trainable_weights 
            if self._exclude_weights(w.name, self.cfg.excluding_weight_names) and "kernel" in w.name
        ]) * self.cfg.weight_decay

        loss = tf.add_n([v for k, v in losses.items() if "loss" in k]) + [l2_loss]
            
        for key, value in losses.items():
            if "loss" not in key: continue
            if key not in self.val_loss_metrics:
                self.val_loss_metrics[key] = tf.keras.metrics.Mean()
            
            self.val_loss_metrics[key].update_state(value)
        
        self.val_loss_metrics["loss"].update_state(loss)
        self.val_loss_metrics["l2_loss"].update_state(l2_loss)
        x = self.detector.head.get_boxes(outputs)
        if tf.equal(self.val_steps.value() % 50, 0):
            normalized_factor = tf.convert_to_tensor(
                [tf.shape(images)[1], tf.shape(images)[2]] * 2, tf.float32)
            gt_boxes = image_info["boxes"]
            pred_boxes = x["nmsed_boxes"]
            gt_boxes = tf.concat([
                gt_boxes[..., 1:2], 
                gt_boxes[..., 0:1], 
                gt_boxes[..., 3:4], 
                gt_boxes[..., 2:3]], 
                -1) / normalized_factor
            pred_boxes = tf.concat([
                pred_boxes[..., 1:2], 
                pred_boxes[..., 0:1], 
                pred_boxes[..., 3:4], 
                pred_boxes[..., 2:3]], 
                -1) / normalized_factor
            images = tf.cast(images, tf.float32)
            images = tf.image.draw_bounding_boxes(images, gt_boxes, tf.constant([[255., 0., 0.]]))
            images = tf.image.draw_bounding_boxes(images, pred_boxes, tf.constant([[0., 255., 0.]]))
            images = tf.cast(images, tf.uint8)

            with tf.device("/cpu:0"):
                with self.summary_writer.as_default():
                    tf.summary.image("val/images", images, self.val_steps.value(), 5)

        return image_info["boxes"], image_info["labels"], x["nmsed_boxes"], x["nmsed_scores"], x["nmsed_classes"] + 1 

    def _validate(self):
        # VAL LOOP
        tf.print("=" * 150)
        val_start = time.time()
        for images, image_info in self.val_dataset:
            self.val_steps.assign_add(1)
            gt_boxes, gt_labels, pred_boxes, pred_scores, pred_classes = self.val_step(
                images, image_info)
            self.ap_metric.update_state(gt_boxes, gt_labels, pred_boxes, pred_scores, pred_classes)
        
        info = "VAL %d: "
        info_values = [self.global_step.numpy()]
        with self.summary_writer.as_default():
            for key in self.val_loss_metrics:
                result = self.val_loss_metrics[key].result()
                self.val_loss_metrics[key].reset_states()
                tf.summary.scalar("val/" + key, result, self.global_step)
                info += key + " = %f "
                info_values.append(result.numpy())
            ap = self.ap_metric.result()
            val_end = time.time()

            info += "\n"
            for i in range(10):
                n = 50 + i * 5
                v = tf.reduce_mean(ap[:, i])
                tf.summary.scalar("val/ap%d" % n, v, step=self.global_step)
                info += " ap%d = %f "
                info_values.append(n)
                info_values.append(v.numpy())
            
            map_ = tf.reduce_mean(ap)
            tf.summary.scalar("val/map", map_, step=self.global_step)
            info += " map = %f "
            info_values.append(map_.numpy())
            self.ap_metric.reset_states()
        info += "(%.2fs)" % (val_end - val_start)
        self.logger.info(info, *info_values)

    def _train_one_epoch(self):
        start = time.time()
        tf.print("=" * 150)
        count = 0
        for images, image_info in self.train_dataset:
            self.global_step.assign_add(1)
            count += 1

            if self._add_graph:
                tf.summary.trace_on(graph=True, profiler=False)
                self.train_step(images, image_info)
                with self.summary_writer.as_default():
                    tf.summary.trace_export(name=self.cfg.detector, step=0)
                self._add_graph = False
            else:
                self.train_step(images, image_info)

            info = "TRAIN %d: "
            info_values = [self.global_step.value().numpy()]
            if tf.equal(self.global_step % self.log_every_n_steps, 0):
                with self.summary_writer.as_default():
                    for key in self.train_loss_metrics:
                        value = self.train_loss_metrics[key].result()
                        tf.summary.scalar("train/" + key, value, self.global_step)
                        info += key + " = %f "
                        info_values.append(value.numpy())
                        self.train_loss_metrics[key].reset_states()
                    
                    lr = self.learning_rate_scheduler(self.global_step.value())
                    tf.summary.scalar("learning_rate", lr, self.global_step)
                    info += "lr = %f(%.2fs)"
                    info_values.append(lr.numpy())
                    info_values.append((time.time() - start) / count)
                self.logger.info(info, *info_values)
                assert all([np.logical_not(np.isnan(v)) for v in info_values]), print(info_values)
                start = time.time()
                count = 0                  
                        
    def run(self):
        # TRAIN LOOP
        for _ in range(self.epochs, self.cfg.train.scheduler.train_epochs):
            self._train_one_epoch()
            self._validate()
            self.manager.save(self.global_step)
            self.logger.info("Saved checkpoints to {}.".format(self.manager.latest_checkpoint))
       
        self.summary_writer.close()
        self.logger.info("Training over.")
        
