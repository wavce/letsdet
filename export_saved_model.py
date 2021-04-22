import argparse
import tensorflow as tf 
from models import build_detector
from configs import build_configs
from core import build_optimizer


parser = argparse.ArgumentParser()
parser.add_argument("--detector", required=True, type=str)
parser.add_argument("--config", type=str, default=None, help="The yaml file, default None.")
parser.add_argument("--saved_model_dir", required=True, default=None, type=str)
parser.add_argument("--ckpt", type=str, default=None, help="The checkpoint dir or h5 file.")

parser.add_argument("--nms", type=str, default="CombinedNonMaxSuppression", help="The NMS type.")
parser.add_argument("--iou_threshold", type=float, default=0.5, help="The iou threshold for NMS.")
parser.add_argument("--score_threshold", type=float, default=0.3, help="The score threshold for NMS.")
parser.add_argument("--update_threshold", type=float, default=0.1, help="The update threshold for MatrixNMS.")
parser.add_argument("--pre_nms_size", type=int, default=4000, help="The number of detections before NMS.")
parser.add_argument("--post_nms_size", type=int, default=100, help="The number of detections after NMS.")
parser.add_argument("--nms_kernel", default="gaussian", type=str, help="The kernel type of MatrixNMS.")
parser.add_argument("--nms_sigma", default=2.0, type=float, help="The sigma for MatrixNMS or SoftNMS.")
parser.add_argument("--nms_type", type=str, default=None, 
                    help="If [--nms] is NonMaxSuppressionWithQuality, the [--nms_type] is necessary.")

args = parser.parse_args()


cfg = build_configs(args.detector)

if args.config is None:
    cfg.test.nms = args.nms
    cfg.test.iou_threshold = args.iou_threshold
    cfg.test.score_threshold = args.score_threshold
    cfg.test.pre_nms_size = args.pre_nms_size
    cfg.test.post_nms_size = args.post_nms_size

    if args.nms == "MatrixNonMaxSuppression":
        cfg.test.update_threshold = args.update_threshold
        cfg.test.kernel = args.nms_kernel

    if args.nms == "NonMaxSuppressionWithQuality":
        assert args.nms_type is not None, "When [--nms] is `NonMaxSuppressionWithQuality`, [--nms_type] is necessary."

    if args.nms in ["MatrixNonMaxSuppression", "SoftNonMaxSuppression"]:
        cfg.test.sigma = args.nms_sigma

    if args.nms == "NonMaxSuppressionWithQuality":
        cfg.test.nms_type = args.nms_type
        if args.nms_type in ["soft_nms", "matrix_nms"]:
            cfg.test.sigma = args.nms_sigma
else:
    cfg.override(args.config)

detector = build_detector(cfg.detector, return_loss=False, cfg=cfg)
images = tf.random.uniform([1, cfg.train.input_size[0], cfg.train.input_size[1], 3])
images = tf.cast(images, tf.uint8)
detector(images)

if args.ckpt is not None and ".h5" in args.ckpt:
    detector.load_weights(args.ckpt)
else:
    optimizer = build_optimizer(**cfg.train.optimizer.as_dict())

    checkpoint = tf.train.Checkpoint(optimizer=optimizer, detector=detector)
    manager = tf.train.CheckpointManager(
        checkpoint=checkpoint, directory=cfg.train.checkpoint_dir, max_to_keep=10)
    latest_checkpoint = manager.latest_checkpoint
    checkpoint.restore(latest_checkpoint)


saved_model_dir = args.saved_model_dir or "./saved_model/" + args.detector

tf.saved_model.save(detector, saved_model_dir)
print("saved model to %s" % saved_model_dir)

# images = tf.random.uniform([1, cfg.train.input_size[0], cfg.train.input_size[1], 3])
# image_info = {"valid_size": tf.constant([[cfg.train.input_size[0], cfg.train.input_size[1]]]), 
#               "input_size": tf.constant([[cfg.train.input_size[0], cfg.train.input_size[1]]]), 
#               "scale_factor": 1.}
# print(detector((images, image_info), training=False))
