import time
import math
import tensorflow as tf
from core import metrics
from data import build_dataset
from core import build_optimizer
from models import build_detector
from core import build_learning_rate_scheduler


class MultiGPUTrainer(object):
    def __init__(self, cfg, logger):
        self.logger = logger
        self.strategy = tf.distribute.MirroredStrategy()
        self.num_replicas = self.strategy.num_replicas_in_sync
        self.train_batch_size = self.num_replicas * cfg.train.dataset.batch_size 
        self.val_batch_size = self.num_replicas * cfg.val.dataset.batch_size

        use_mixed_precision = cfg.dtype in ["FP16", "float16"]
        cfg.train.dataset.batch_size = self.train_batch_size
        cfg.val.dataset.batch_size = self.val_batch_size
        train_dataset = build_dataset(
            dtype=tf.float16 if use_mixed_precision else tf.float32,
            **cfg.train.dataset.as_dict())
        val_dataset = build_dataset(
            dtype=tf.float16 if use_mixed_precision else tf.float32,
            **cfg.val.dataset.as_dict())

        with self.strategy.scope():
            self.train_dataset = self.strategy.experimental_distribute_dataset(train_dataset)
            self.val_dataset = self.strategy.experimental_distribute_dataset(val_dataset)

            if use_mixed_precision:
                tf.keras.mixed_precision.set_global_policy("mixed_float16") 
                print("Using mixed precision training.")

            if cfg.train.get("proposal_layer"):
                self.detector = build_detector(cfg.detector, cfg=cfg, proposal_cfg=cfg.train.proposal_layer)
            else:
                self.detector = build_detector(cfg.detector, cfg=cfg)
                
            self.detector.load_pretrained_weights(cfg.train.pretrained_weights_path)

            train_steps_per_epoch = cfg.train.dataset.num_samples // cfg.train.dataset.batch_size
            self.total_train_steps = cfg.train.scheduler.train_epochs * train_steps_per_epoch
            self.learning_rate_scheduler = build_learning_rate_scheduler(
                **cfg.train.scheduler.learning_rate_scheduler.as_dict(), 
                train_steps=self.total_train_steps, 
                warmup_learning_rate=cfg.train.scheduler.warmup.warmup_learning_rate,
                warmup_steps=cfg.train.scheduler.warmup.steps,
                train_steps_per_epoch=train_steps_per_epoch)
            optimizer = build_optimizer(learning_rate=self.learning_rate_scheduler, **cfg.train.optimizer.as_dict())

            if use_mixed_precision:
                optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer, dynamic=True) 

            self.optimizer = optimizer
            self.use_mixed_precision = use_mixed_precision
            self.cfg = cfg
            
            self.global_step = tf.Variable(
                initial_value=0, 
                trainable=False, 
                name="global_step", 
                dtype=tf.int64)

            self.val_steps = tf.Variable(0, trainable=False, name="val_steps", dtype=tf.int64)
                    
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
                self.logger.info("Restored weights from %s." % latest_checkpoint)
            else:
                self.global_step.assign(0)

            self.summary_writer = tf.summary.create_file_writer(logdir=cfg.train.summary_dir)
            self.log_every_n_steps = cfg.train.log_every_n_steps
            self.use_jit = tf.config.optimizer.get_jit() is not None

            self.train_loss_metrics = {"l2_loss": tf.keras.metrics.Mean(), "loss": tf.keras.metrics.Mean()}
            self.val_loss_metrics = {"l2_loss": tf.keras.metrics.Mean(), "loss": tf.keras.metrics.Mean()}
            self._add_graph = True
            self.ap_metric = metrics.mAP(self.cfg.num_classes)

    def run(self):
        with self.strategy.scope():
            def _exclude_weights(name, excluding_weight_names=[]):
                for n in excluding_weight_names:
                    if n in name:
                        return False
                
                return True

            def _train_step_fn(images, image_info):
                with tf.GradientTape() as tape:
                    outputs = self.detector.detector(images, training=True)
                    losses = self.detector.compute_losses(outputs, image_info)   

                    l2_loss =  tf.add_n([
                        tf.nn.l2_loss(w) for w in self.detector.detector.trainable_weights 
                        if _exclude_weights(w.name, self.cfg.excluding_weight_names) and "kernel" in w.name
                    ]) * self.cfg.weight_decay
                    
                    loss = tf.add_n([v for k, v in losses.items()] + [l2_loss]) 
                    replica_loss = loss / tf.cast(self.num_replicas, loss.dtype)
                    if self.use_mixed_precision:
                        scaled_loss = self.optimizer.get_scaled_loss(replica_loss)
                    else:
                        scaled_loss = replica_loss

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
                   
                    images = tf.cast(images, tf.float32)
                    images = tf.image.draw_bounding_boxes(images, gt_boxes, tf.constant([[255., 0., 0.]]))
                    images = tf.image.draw_bounding_boxes(images, pred_boxes, tf.constant([[0., 255., 0.]]))
                    images = tf.cast(images, tf.uint8)
                    with tf.device("/cpu:0"):
                        with self.summary_writer.as_default():
                            tf.summary.image("train/images", images, self.global_step, 5)

                return loss

            def _test_step_fn(images, image_info):
                outputs = self.detector.detector(images, training=True)
                losses = self.detector.compute_losses(outputs, image_info)

                l2_loss = tf.add_n([
                    tf.nn.l2_loss(w) for w in self.detector.detector.trainable_weights 
                    if _exclude_weights(w.name, self.cfg.excluding_weight_names) and "kernel" in w.name
                ]) * self.cfg.weight_decay
                    
                loss = tf.add_n([v for k, v in losses.items()] + [l2_loss])

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

            @tf.function(experimental_relax_shapes=True, input_signature=self.train_dataset.element_spec)
            def distributed_train_step(images, image_info):
                per_replica_losses = self.strategy.run(_train_step_fn, args=(images, image_info))
                return self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

            @tf.function(experimental_relax_shapes=True, input_signature=self.val_dataset.element_spec)
            def distributed_valuate_step(images, image_info):
                return self.strategy.run(_test_step_fn, args=(images, image_info))
            
            def _train_one_epoch():
                # TRAIN LOOP
                count = 0
                start = time.time()
                for images, image_info in self.train_dataset:
                    self.global_step.assign_add(1)
                    if self._add_graph:
                        tf.summary.trace_on(graph=True, profiler=False)
                        distributed_train_step(images, image_info)

                        with self.summary_writer.as_default():
                            tf.summary.trace_export(name=self.cfg.detector, step=0)
                        self._add_graph = False
                    else:
                        distributed_train_step(images, image_info)

                    count += 1

                    info = "TRAIN %d:"
                    info_values =[self.global_step.numpy()]
                    if tf.equal(self.global_step % self.log_every_n_steps, 0):
                        with self.summary_writer.as_default():
                            for key in self.train_loss_metrics:
                                value = self.train_loss_metrics[key].result()
                                tf.summary.scalar("train/" + key, value, self.global_step)
                                info += " " + key + " = %f "
                                info_values.append(value.numpy())
                                
                                self.train_loss_metrics[key].reset_states()
                            
                            lr = self.learning_rate_scheduler(self.global_step.value())
                            tf.summary.scalar("learning_rate", lr, self.global_step)
                            info += "lr = %f(%.2fs)"
                            info_values.append(lr.numpy())
                            info_values.append((time.time() - start) / count)
                        self.logger.info(info, *info_values)
                        start = time.time()
                        count = 0

            def _validate():
                print("=" * 150)
                # EVALUATING LOO
                val_start = time.time()
                for images, image_info in self.val_dataset:
                    self.val_steps.assign_add(1)
                    # distributed_valuate_step(images, image_info)
                    gt_boxes, gt_labels, pred_boxes, pred_scores, pred_classes = distributed_valuate_step(
                        images, image_info)

                    gt_boxes = [b for x in tf.nest.flatten(gt_boxes) for b in self.strategy.unwrap(x)]
                    gt_labels = [l for x in tf.nest.flatten(gt_labels) for l in self.strategy.unwrap(x)]
                    pred_boxes = [b for x in tf.nest.flatten(pred_boxes) for b in self.strategy.unwrap(x)]
                    pred_scores = [s for x in tf.nest.flatten(pred_scores) for s in self.strategy.unwrap(x)]
                    pred_classes = [c for x in tf.nest.flatten(pred_classes) for c in self.strategy.unwrap(x)]
                    gt_boxes = tf.concat(gt_boxes, 0)
                    gt_labels = tf.concat(gt_labels, 0)
                    pred_boxes = tf.concat(pred_boxes, 0)
                    pred_scores = tf.concat(pred_scores, 0)
                    pred_classes = tf.concat(pred_classes, 0)

                    self.ap_metric.update_state(gt_boxes, gt_labels, pred_boxes, pred_scores, pred_classes)

                val_end = time.time()
                info = "VAL %d"
                info_values =[self.global_step.value().numpy()]
                with self.summary_writer.as_default():
                    for name in self.val_loss_metrics:
                        value = self.val_loss_metrics[name].result()
                        tf.summary.scalar("val/" + name, value, self.global_step)
                        info += " %s = %f"
                        info_values.append(name) 
                        info_values.append(value)
                        self.val_loss_metrics[name].reset_states()

                    ap = self.ap_metric.result()
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
                    self.ap_metric.reset_states()
                    info += " ap = %f(%.2fs)"
                    info_values.append(map_.numpy()) 
                    info_values.append(val_end - val_start)
                self.logger.info(info, *info_values)

        # TRAIN LOOP
        for _ in range(self.epochs, self.cfg.train.scheduler.train_epochs):
            _train_one_epoch()
            _validate()
            self.manager.save(self.global_step)
            self.logger.info("Saved checkpoints to {}.".format(self.manager.latest_checkpoint))
        
        self.logger.info("Training over.")
        self.summary_writer.close()
