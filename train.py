import os
import logging
import argparse
import tensorflow as tf
from configs import Config
from configs import build_configs
from trainers import MultiGPUTrainer
from trainers import SingleGPUTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", 
                        type=str,
                        default="CenterNet",
                        help="The detector name, e.g.`efficientdet`, `efficient_fcos`.")
    parser.add_argument("--gpus", 
                        type=str,
                        default="0,1,2,3",
                        help="Use multi-gpu training or not, default False, means use one gpu.")
    parser.add_argument("--cfg",
                        type=str,
                        default=None,
                        help="The conifg file (yaml), if None, using default.")
    parser.add_argument("--num_classes",
                        type=int,
                        default=80,
                        help="The number of classes, default 80 (COCO).")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    tf.random.set_seed(2333)
    # tf.config.optimizer.set_jit(True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logging.basicConfig(format="%(asctime)s %(levelname)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")

    physical_devices = tf.config.experimental.list_physical_devices("GPU")
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

    if args.cfg is None:
        cfg = build_configs(args.detector)(args.num_classes)
    else:
        cfg = Config()
        cfg.parse_from_yaml(args.cfg)

    num_gpus = len(args.gpus.strip().split(","))
    if num_gpus > 1:
        trainer = MultiGPUTrainer(cfg=cfg, logger=logger)
    else:
        trainer = SingleGPUTrainer(cfg=cfg, logger=logger)
    
    trainer.run()


if __name__ == '__main__':
    main()
