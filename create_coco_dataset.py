import cv2
import argparse
from data.datasets.coco_dataset import COCODataset


parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", default=None, type=str,
                    help="""The directory contains image and anntotation filePycharmProjects:
                            │  └─COCO
                            │      ├─train2017(training images)
                            │      ├─val2017(valimages)
                            │      └─annotations""")
parser.add_argument("--phase", default="train", type=str, 
                    help="The phase of dataset, e.g. for `train2017`, the value should be `train`."
                            " for `val2017`, the value should be `val`.")
parser.add_argument("--version", default=2017, type=int, 
                    help="The version of dataset, e.g. for `train2017`, the value should be `2017`," 
                            " for `val2017`, the value should be `2017`.")
parser.add_argument("--max_images_per_tfrecord", default=20000, type=int,
                    help="The maximum images per tfrecord.")

args = parser.parse_args()

assert args.dataset_dir is not None, "Must provide dataset directory."

coco = COCODataset(args.dataset_dir, training=True)
coco.create_tf_record(phase=args.phase, version=args.version, max_imgs_per_tfrecord=args.max_images_per_tfrecord)
    