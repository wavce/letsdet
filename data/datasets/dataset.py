import math
import tensorflow as tf
from ..augmentations import Mixup
from ..augmentations import Mosaic
from ..builder import build_augmentation


class Dataset(object):
    def __init__(self, 
                 dataset_dir, 
                 training=True,
                 batch_size=32, 
                 augmentations=[],
                 max_boxes=200,
                 skip_crowd=True,
                 mosaic=None,
                 mixup=None,
                 dtype=tf.float32,
                 **kwargs):
        self.dataset_dir = dataset_dir
        self.training = training
        self.batch_size = batch_size
        self.max_boxes = max_boxes
        self.skip_crowd = skip_crowd
        self.dtype = dtype
        
        if mosaic is not None:
            self.mosaic = Mosaic(max_boxes=max_boxes, **mosaic)
            assert "ResizeV2" in [list(n.keys())[0] for n in augmentations], "Whe using Mosaic, ResizeV2 shoud in augmentations."
        if mixup is not None:
            self.mixup = Mixup(batch_size=batch_size, max_boxes=max_boxes, **mixup)
        
        self.augmentations = [build_augmentation(**kw) for kw in augmentations]
    
    def compose(self, image, image_info):
        for aug in self.augmentations:
            image, image_info = aug(image, image_info)
        
        return image, image_info

    def is_valid_jpg(self, jpg_file):
        with open(jpg_file, 'rb') as f:
            f.seek(-2, 2)
            buf = f.read()
            f.close()
            return buf == b'\xff\xd9'  # 判定jpg是否包含结束字段
    
    def _bytes_list_feature(self, value):
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))
    
    def _int64_list_feature(self, value):
        if not isinstance(value, list):
            value = [value]
        
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
    
    def _float_list_feature(self, value):
        if not isinstance(value, list):
            value = [value]
        
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    def create_tfrecord(self, image_dir, image_info_file, output_dir, num_shards):
        raise NotImplementedError()

    def parser(self, serialized):
        raise NotImplementedError()

    def dataset(self):
        with tf.device("/cpu:0"):
            dataset = tf.data.TFRecordDataset(self.tf_record_sources)
            dataset = dataset.map(map_func=self.parser)
            
            if hasattr(self, "mosaic"):
                self.batch_size *= 4
            if self.training:
                dataset = dataset.shuffle(buffer_size=self.batch_size * 10)
            dataset = dataset.batch(batch_size=self.batch_size, drop_remainder=True)
            
            if hasattr(self, "mosaic"):
                dataset = dataset.map(self.mosaic)
            
            # call mixup shoud after mosaic
            if hasattr(self, "mixup"):
                dataset = dataset.map(self.mixup)
           
            return dataset.prefetch(tf.data.experimental.AUTOTUNE)