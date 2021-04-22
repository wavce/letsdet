import os
import io
import glob
import tqdm
import numpy as np
import tensorflow as tf 
import PIL.Image as Image
from pycocotools import mask
from pycocotools.coco import COCO
from .dataset import Dataset
from ..builder import DATASETS


@DATASETS.register
class COCODataset(Dataset):
    def __init__(self, 
                 dataset_dir, 
                 training=True,
                 batch_size=32, 
                 include_masks=False,
                 augmentations=[],
                 mosaic=None,
                 mixup=None,
                 dtype=tf.float32,
                 **kwargs):
        super(COCODataset, self).__init__(dataset_dir=dataset_dir, 
                                          training=training, 
                                          batch_size=batch_size, 
                                          augmentations=augmentations,
                                          mosaic=mosaic,
                                          mixup=mixup,
                                          dtype=dtype,
                                          **kwargs)
        self.include_masks = include_masks
    
    def create_tf_record(self, phase="train", version=2017, max_imgs_per_tfrecord=20000):
        annotation_path = os.path.join(self.dataset_dir, "annotations", "instances_{}{}.json".format(phase, version))
        if not os.path.exists(annotation_path):
            raise FileNotFoundError("Not Found [{}].".format(annotation_path))

        coco = COCO(annotation_path)
    
        img_ids = coco.getImgIds()
        cat_ids = coco.getCatIds()

        num_imgs = len(img_ids)
        num_tfrecords = num_imgs // max_imgs_per_tfrecord 
        if num_imgs % max_imgs_per_tfrecord != 0:
            num_tfrecords += 1

        writer = None
        for i, img_id in tqdm.tqdm(enumerate(img_ids)):
            if i % max_imgs_per_tfrecord == 0:
                if writer is not None:
                    writer.close()
                tfrecord_name = "{}{}-{:05}-{:05}.tfrec".format(phase, version, i // max_imgs_per_tfrecord + 1, num_tfrecords)
                tfrecord_path = os.path.join(self.dataset_dir, tfrecord_name)
                writer = tf.io.TFRecordWriter(tfrecord_path)

            img_info = coco.loadImgs(img_ids[i])[0]
            img_height = img_info["height"]
            img_width = img_info["width"]
            img_path = os.path.join(self.dataset_dir, "{}{}".format(phase, version), img_info["file_name"])

            ann_ids = coco.getAnnIds(imgIds=img_info["id"])
            anns = coco.loadAnns(ann_ids)
            
            iscrowds = []
            xmins = []
            ymins = []
            xmaxs = []
            ymaxs = []
            category_ids = []
            categories = []
            category_names = []
            encoded_mask_png = []
            for ann in anns:
                iscrowds.append(ann["iscrowd"])
                xmins.append(ann["bbox"][0])
                ymins.append(ann["bbox"][1])
                xmaxs.append(ann["bbox"][0] + ann["bbox"][2])
                ymaxs.append(ann["bbox"][1] + ann["bbox"][3])
                
                cat_id = ann["category_id"]
                category_ids.append(cat_id)
                label = cat_ids.index(cat_id) + 1
                categories.append(label)
                run_len_encoding = mask.frPyObjects(ann["segmentation"], img_height, img_width)
            
                binary_mask = mask.decode(run_len_encoding)
                if not ann["iscrowd"]:
                    binary_mask = np.amax(binary_mask, axis=2)
                
                pil_img = Image.fromarray(binary_mask)
                output_io = io.BytesIO()
                pil_img.save(output_io, format="PNG")
                encoded_mask_png.append(output_io.getvalue())

                cat_info = coco.loadCats(ann["category_id"])[0]
                category_names.append(cat_info["name"].encode("utf-8"))

            with tf.io.gfile.GFile(img_path, "rb") as gf:
                raw = gf.read()
            
            example = tf.train.Example(
                features=tf.train.Features(feature={
                    "image/encoded": self._bytes_list_feature(raw),
                    "image/height": self._int64_list_feature(img_height),
                    "image/width": self._int64_list_feature(img_width),
                    "image/objects/count": self._int64_list_feature(len(anns)),
                    "image/objects/x1": self._float_list_feature(xmins),
                    "image/objects/y1": self._float_list_feature(ymins),
                    "image/objects/x2": self._float_list_feature(xmaxs),
                    "image/objects/y2": self._float_list_feature(ymaxs),
                    "image/objects/categories": self._int64_list_feature(categories),
                    "image/objects/category_ids": self._int64_list_feature(category_ids),
                    "image/objects/category_names": self._bytes_list_feature(category_names),
                    "image/objects/iscrowds": self._int64_list_feature(iscrowds),
                    "image/objects/mask": self._bytes_list_feature(encoded_mask_png)
                }))
            
            writer.write(example.SerializeToString())

        writer.close()
    
    def parse(self, serialized):
        key_to_features = {
            "image/encoded": tf.io.FixedLenFeature([], tf.string, ""),
            "image/height": tf.io.FixedLenFeature([], tf.int64, 0),
            "image/width": tf.io.FixedLenFeature([], tf.int64, 0),
            "image/objects/count": tf.io.FixedLenFeature([], tf.int64, 0),
            "image/objects/x1": tf.io.VarLenFeature(tf.float32),
            "image/objects/y1": tf.io.VarLenFeature(tf.float32),
            "image/objects/x2": tf.io.VarLenFeature(tf.float32),
            "image/objects/y2": tf.io.VarLenFeature(tf.float32),
            "image/objects/iscrowds": tf.io.VarLenFeature(tf.int64),
            "image/objects/categories": tf.io.VarLenFeature(tf.int64),
            "image/objects/mask": tf.io.VarLenFeature(tf.string)
        }

        features = tf.io.parse_single_example(serialized, key_to_features)
        image = tf.image.decode_image(features["image/encoded"], channels=3, dtype=tf.uint8)
        height = features["image/height"]
        width = features["image/width"]
        image = tf.reshape(image, [height, width, 3], name="image_reshape")
        image = tf.cast(image, tf.float32)
        count = features["image/objects/count"]
        x1 = tf.sparse.to_dense(features["image/objects/x1"])
        y1 = tf.sparse.to_dense(features["image/objects/y1"])
        x2 = tf.sparse.to_dense(features["image/objects/x2"])
        y2 = tf.sparse.to_dense(features["image/objects/y2"])
        # x1 = tf.cast(x1, tf.float32) / tf.cast(width, tf.float32)
        # y1 = tf.cast(y1, tf.float32) / tf.cast(height, tf.float32)
        # x2 = tf.cast(x2, tf.float32) / tf.cast(width, tf.float32)
        # y2 = tf.cast(y2, tf.float32) / tf.cast(height, tf.float32)

        boxes = tf.stack([x1, y1, x2, y2], 1)   # (x1, y1, x2, y2)
        iscrowds = tf.sparse.to_dense(features["image/objects/iscrowds"])
        labels = tf.sparse.to_dense(features["image/objects/categories"])
        if self.skip_crowd:
            boxes = tf.boolean_mask(boxes, iscrowds == 0)
            labels = tf.boolean_mask(labels, iscrowds == 0)
        
        image_info = dict(boxes=boxes, labels=labels)

        if self.include_masks:
            raw_masks = tf.sparse.to_dense(features["image/objects/mask"])

            def _decode_mask(raw_mask):
                m = tf.image.decode_png(raw_mask, 1, tf.uint8)
                m = tf.squeeze(m, -1)
                m = tf.reshape(m, [height, width])
                
                return m

            mask = tf.cond(count <= 0, 
                           lambda: tf.zeros([0, height, width], tf.uint8), 
                           lambda: tf.map_fn(_decode_mask, raw_masks, fn_output_signature=tf.uint8))
            image_info["mask"] = mask
        
        image, image_info = self.compose(image, image_info)

        image = tf.cast(image, tf.uint8)
        boxes = image_info["boxes"]
        labels = image_info["labels"]

        num_instances = tf.shape(boxes)[0]
        boxes = tf.cond(num_instances >= self.max_boxes,
                        lambda: boxes[:self.max_boxes],
                        lambda: tf.pad(boxes, [[0, self.max_boxes - num_instances], [0, 0]]))
        labels = tf.cond(num_instances >= self.max_boxes,
                         lambda: labels[:self.max_boxes],
                         lambda: tf.pad(labels, [[0, self.max_boxes - num_instances]]))
        boxes = tf.maximum(boxes, 0.)
        image_info["boxes"] = boxes
        image_info["labels"] = labels

        if self.include_masks:
            mask = image_info["mask"]
            mask = tf.cond(count >= self.max_boxes,
                           lambda: mask[:self.max_boxes],
                           lambda: tf.concat([mask, tf.zeros([self.max_boxes - count, height, width], mask.dtype)], 0))
            image_info["mask"] = mask

        return image, image_info

    def dataset(self):
        with tf.device("/cpu:0"):
            if self.training:
                tfrecord_sources = glob.glob(os.path.join(self.dataset_dir, "train*.tfrec"))
            else:
                tfrecord_sources = glob.glob(os.path.join(self.dataset_dir, "val*.tfrec"))
    
            if len(tfrecord_sources) <= 0:
                raise FileNotFoundError("tfrecord[{}] not found in {}".format("train" if self.training else "val", self.dataset_dir))
            
            ds = tf.data.TFRecordDataset(tfrecord_sources)
            ds = ds.map(self.parse, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            
            batch_size = self.batch_size
            if hasattr(self, "mosaic") and self.training:
                batch_size *= 4

            if self.training:
                ds = ds.shuffle(buffer_size=batch_size * 100)
                # ds = ds.batch(batch_size=batch_size, drop_remainder=True)
                ds = ds.padded_batch(batch_size=batch_size, drop_remainder=True)
            else:
                ds = ds.padded_batch(batch_size=batch_size, drop_remainder=False)
            
            if hasattr(self, "mosaic"):
                ds = ds.map(self.mosaic)
            
            if hasattr(self, "mixup"):
                ds = ds.map(self.mixup)

            return ds.prefetch(tf.data.experimental.AUTOTUNE)
    

def main():
    import cv2
    
    coco = COCODataset("/home/bail/Data/data1/Dataset/COCO", 
                       batch_size=4, 
                       augmentations=[
                        #    dict(augmentation="FlipLeftToRight", probability=0.5),
                        #    dict(augmentation="Resize", img_scale=(0.2, 2), multiscale_mode="range", keep_ratio=True),
                           dict(augmentation="Resize", img_scale=[(1333, 512)], keep_ratio=True),
                           dict(augmentation="Pad", size_divisor=32),
                        #    dict(augmentation="RandCropOrPad", size=(640, 640), clip_box_base_center=False),
                        #    dict(augmentation=ResizeAndCrop, size=(512, 512)),
                        ], 
                    #    mosaic=dict(size=(640, 640), min_image_scale=0.25, prob=1),
                    #    mixup=dict(alpha=8.0),
                       training=False)
    
    cv2.namedWindow("coco", cv2.WINDOW_NORMAL)
    for images, image_info in coco.dataset():
        images = images.numpy()
        boxes = image_info["boxes"].numpy()
        labels = image_info["labels"].numpy()
        # masks = masks.numpy()
    
        for i in range(len(images)):
            img = images[i].astype(np.uint8)
            print(img.shape)
            # img1 = np.concatenate([img[0], img[1]], 0)
            # img2 = np.concatenate([img[2], img[3]], 0)
            # img = np.concatenate([img1, img2], 1)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            # mask = masks[i]
            for j, (box, lbl) in enumerate(zip(boxes[i], labels[i])):
                img = cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255))
                img = cv2.putText(img, str(lbl), (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
  
                # msk = msk[:, :, None] * np.array([[0, 0, 255]], np.uint8)
                # msk[msk == 0] = img[msk == 0]

                # img = cv2.addWeighted(img, 0.7, msk, 0.3, 0)
            cv2.imshow("coco", img)
            cv2.waitKey(0)


if __name__ == "__main__":
    main()
    
