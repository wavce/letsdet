from .base_config import Config
from .yolov5_config import get_yolov5_config
from .atss_config import get_atss_config
from .fcos_config import get_fcos_config
from .faster_rcnn_config import get_faster_rcnn_config
from .efficientdet_config import get_efficientdet_config
from .gfl_config import get_gfl_config
from .centernet_config import get_centernet_config
from .retinanet_config import get_retinanet_config


CONFIG_DICT = {
    "EfficientDetD0": lambda x: get_efficientdet_config("EfficientDetD0", x),
    "EfficientDetD1": lambda x: get_efficientdet_config("EfficientDetD1", x),
    "EfficientDetD2": lambda x: get_efficientdet_config("EfficientDetD2", x),
    "EfficientDetD3": lambda x: get_efficientdet_config("EfficientDetD3", x),
    "EfficientDetD4": lambda x: get_efficientdet_config("EfficientDetD4", x),
    "EfficientDetD5": lambda x: get_efficientdet_config("EfficientDetD5", x),
    "EfficientDetD6": lambda x: get_efficientdet_config("EfficientDetD6", x),
    "EfficientDetD7": lambda x: get_efficientdet_config("EfficientDetD7", x),
    "FasterRCNN": lambda x: get_faster_rcnn_config(x),
    "FCOS": lambda x: get_fcos_config(x),
    "ATSS": lambda x: get_atss_config(x),
    "GFL": lambda x: get_gfl_config(x),
    "YOLOv5s": lambda x: get_yolov5_config(x, .33, .50, "yolov5s"),
    "YOLOv5m": lambda x: get_yolov5_config(x, .67, .75, "yolov5m"),
    "YOLOv5l": lambda x: get_yolov5_config(x, 1., 1., "yolov5l"),
    "YOLOv5x": lambda x: get_yolov5_config(x, 1.22, 1.25, "yolov5x"),
    "CenterNet": lambda x: get_centernet_config(x),
    "RetinaNet": lambda x: get_retinanet_config(x),
    "OneNet": lambda x: get_onenet_config(x),
}


def build_configs(name):
    return CONFIG_DICT[name]
