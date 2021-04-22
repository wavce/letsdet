import time
import tensorflow as tf
from core.bbox import Delta2Box
from .one_stage import OneStageDetector
from models.builder import DETECTORS
from models.builder import build_neck
from models.builder import build_head 
from models.builder import build_backbone


class EfficientDet(OneStageDetector):
    def __init__(self, cfg, return_loss=False, **kwargs):
        super(EfficientDet, self).__init__(cfg, return_loss=return_loss, **kwargs)
    
    def load_pretrained_weights(self, pretrained_weights_path=None):
        pretrained_weights_path = "/home/bail/Workspace/pretrained_weights/efficientdet-d0"
        if "h5" in pretrained_weights_path:
            self.backbone.load_weights(pretrained_weights_path, by_name=True)
            print("Restored pre-trained weights from %s" % pretrained_weights_path)
        elif tf.train.latest_checkpoint(pretrained_weights_path) is not None:
            images = tf.random.uniform([1, self.cfg.train.input_size[0], self.cfg.train.input_size[1], 3])
            images = tf.cast(images, tf.uint8)
            image_info = {"valid_size": tf.constant([[self.cfg.train.input_size[0], self.cfg.train.input_size[1]]]), 
                          "input_size": tf.constant([[self.cfg.train.input_size[0], self.cfg.train.input_size[1]]]), 
                          "scale_factor": 1.}
            self.__call__((images, image_info))
            # self.build(input_shape=(1, self.cfg.train.input_size[0], self.cfg.train.input_size[1], 3))
            self._load_pretraned_weights(pretrained_weights_path)
            print("Restored pre-trained weights from %s" % pretrained_weights_path)
        else:
            print("Train model from scratch.")
    
    def _load_pretraned_weights(self, pretrained_weight_path=None):
        if pretrained_weight_path is not None:
            pretrained_weights = tf.train.latest_checkpoint(pretrained_weight_path)
            assert pretrained_weights is not None, "Error! Please check path {}".format(pretrained_weight_path)
            use_exponential_moving_average = False
            # for w in tf.train.list_variables(pretrained_weights):
            #     if "ExponentialMovingAverage" not in w[0]:
            #         # use_exponential_moving_average = True
            #         # if "box_net" in w[0] or "class_net" in w[0]:
            #              print(w[0])

            for weight in self.weights:
                name = weight.name.split(":")[0]
                if "box_net" in name or "class_net" in name:
                    name = name.split("retina_net_head/")[1]
                    # print(name)
                
                # if "box-predict" in name or "class-predict" in name:
                #     continue
                if "batch_normalization" in name:
                    name = name.replace("batch_normalization", "tpu_batch_normalization")
                
                if use_exponential_moving_average:
                    name += "/ExponentialMovingAverage"
                try:
                    pretrained_weight = tf.train.load_variable(pretrained_weights, name)
                    weight.assign(pretrained_weight)
                except Exception as e:
                    print(str(e), ", {} not in {}.".format(name, pretrained_weight_path))
                    pass


@DETECTORS.register
class EfficientDetD0(EfficientDet):
    def __init__(self, cfg, **kwargs):
        super(EfficientDetD0, self).__init__(cfg, **kwargs)


@DETECTORS.register
class EfficientDetD1(EfficientDet):
    def __init__(self, cfg, **kwargs):
        super(EfficientDetD1, self).__init__(cfg, **kwargs)


@DETECTORS.register
class EfficientDetD2(EfficientDet):
    def __init__(self, cfg, **kwargs):
        super(EfficientDetD2, self).__init__(cfg, **kwargs)


@DETECTORS.register
class EfficientDetD3(EfficientDet):
    def __init__(self, cfg, **kwargs):
        super(EfficientDetD3, self).__init__(cfg, **kwargs)


@DETECTORS.register
class EfficientDetD4(EfficientDet):
    def __init__(self, cfg, **kwargs):
        super(EfficientDetD4, self).__init__(cfg, **kwargs)


@DETECTORS.register
class EfficientDetD5(EfficientDet):
    def __init__(self, cfg, **kwargs):
        super(EfficientDetD5, self).__init__(cfg, **kwargs)


@DETECTORS.register
class EfficientDetD6(EfficientDet):
    def __init__(self, cfg, **kwargs):
        super(EfficientDetD6, self).__init__(cfg, **kwargs)


@DETECTORS.register
class EfficientDetD7(EfficientDet):
    def __init__(self, cfg, **kwargs):
        super(EfficientDetD7, self).__init__(cfg, **kwargs)
