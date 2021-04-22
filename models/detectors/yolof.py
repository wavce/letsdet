import tensorflow as tf 
from .one_stage import OneStageDetector
from ..builder import DETECTORS


@DETECTORS.register
class YOLOF(OneStageDetector):
    def __init__(self, cfg, **kwargs):
        super(YOLOF, self).__init__(cfg=cfg, **kwargs)



_HEAD_WEIGHT_MAP = {
    "head.cls_subnet.0.conv2d.weight":  "decoder.cls_subnet.0.weight",
    "head.cls_subnet.0.conv2d.bias":  "decoder.cls_subnet.0.bias",
    "head.cls_subnet.0.norm.weight":  "decoder.cls_subnet.1.weight",
    "head.cls_subnet.0.norm.bias":  "decoder.cls_subnet.1.bias",
    "head.cls_subnet.0.norm.running_mean":  "decoder.cls_subnet.1.running_mean",
    "head.cls_subnet.0.norm.running_var":  "decoder.cls_subnet.1.running_var",
    "head.cls_subnet.1.conv2d.weight":  "decoder.cls_subnet.3.weight",
    "head.cls_subnet.1.conv2d.bias":  "decoder.cls_subnet.3.bias",
    "head.cls_subnet.1.norm.weight":  "decoder.cls_subnet.4.weight",
    "head.cls_subnet.1.norm.bias":  "decoder.cls_subnet.4.bias",
    "head.cls_subnet.1.norm.running_mean":  "decoder.cls_subnet.4.running_mean",
    "head.cls_subnet.1.norm.running_var":  "decoder.cls_subnet.4.running_var",
    "head.bbox_subnet.0.conv2d.weight":  "decoder.bbox_subnet.0.weight",
    "head.bbox_subnet.0.conv2d.bias":  "decoder.bbox_subnet.0.bias",
    "head.bbox_subnet.0.norm.weight":  "decoder.bbox_subnet.1.weight",
    "head.bbox_subnet.0.norm.bias":  "decoder.bbox_subnet.1.bias",
    "head.bbox_subnet.0.norm.running_mean":  "decoder.bbox_subnet.1.running_mean",
    "head.bbox_subnet.0.norm.running_var":  "decoder.bbox_subnet.1.running_var",
    "head.bbox_subnet.1.conv2d.weight":  "decoder.bbox_subnet.3.weight",
    "head.bbox_subnet.1.conv2d.bias":  "decoder.bbox_subnet.3.bias",
    "head.bbox_subnet.1.norm.weight":  "decoder.bbox_subnet.4.weight",
    "head.bbox_subnet.1.norm.bias":  "decoder.bbox_subnet.4.bias",
    "head.bbox_subnet.1.norm.running_mean":  "decoder.bbox_subnet.4.running_mean",
    "head.bbox_subnet.1.norm.running_var":  "decoder.bbox_subnet.4.running_var",
    "head.bbox_subnet.2.conv2d.weight":  "decoder.bbox_subnet.6.weight",
    "head.bbox_subnet.2.conv2d.bias":  "decoder.bbox_subnet.6.bias",
    "head.bbox_subnet.2.norm.weight":  "decoder.bbox_subnet.7.weight",
    "head.bbox_subnet.2.norm.bias":  "decoder.bbox_subnet.7.bias",
    "head.bbox_subnet.2.norm.running_mean":  "decoder.bbox_subnet.7.running_mean",
    "head.bbox_subnet.2.norm.running_var":  "decoder.bbox_subnet.7.running_var",
    "head.bbox_subnet.3.conv2d.weight":  "decoder.bbox_subnet.9.weight",
    "head.bbox_subnet.3.conv2d.bias":  "decoder.bbox_subnet.9.bias",
    "head.bbox_subnet.3.norm.weight":  "decoder.bbox_subnet.10.weight",
    "head.bbox_subnet.3.norm.bias":  "decoder.bbox_subnet.10.bias",
    "head.bbox_subnet.3.norm.running_mean":  "decoder.bbox_subnet.10.running_mean",
    "head.bbox_subnet.3.norm.running_var":  "decoder.bbox_subnet.10.running_var",
}


def _load_weight_from_torch(model, torch_weights_path):
    import re
    import torch
    import numpy as np
    import torch.nn as nn

    # model.summary()
    pretrained = torch.load(torch_weights_path, map_location=torch.device("cpu"))["model"]
    # for k, v in pretrained.items():
    #     if"tracked" not in k:
    #         print(k, v.shape)
    
    for weight in model.weights:
        name = weight.name
        # print(name, weight.shape)
        name = name.split(":")[0]
        name = name.replace("/", ".")

        if "stem" in name:
            name = "backbone." + name

        if "layer" in name:
            name = name.replace("layer1", "res2")
            name = name.replace("layer2", "res3")
            name = name.replace("layer3", "res4")
            name = name.replace("layer4", "res5")
            name = "backbone." + name
        
        if "batch_norm" in name:
            name = name.replace("batch_norm", "norm")

        if "conv2d.kernel" in name and "head" not in name:
            name = name.replace("conv2d.kernel", "weight")
        if "conv2d.bias" in name and "head" not in name:
            name = name.replace("conv2d.bias", "bias")
        if "kernel" in name and "head" not in name:
            name = name.replace("kernel", "weight")
        
        if "gamma" in name:
            name = name.replace("gamma", "weight")
        if "beta" in name:
            name = name.replace("beta", "bias")
        if "moving_mean" in name:
            name = name.replace("moving_mean", "running_mean")
        if "moving_variance" in name:
            name = name.replace("moving_variance", "running_var")
    
        if "conv1.batch_norm" in name:
            name = name.replace("conv1.batch_norm", "bn1")
        if "conv2.batch_norm" in name:
            name = name.replace("conv2.batch_norm", "bn2")
        if "conv3.batch_norm" in name:
            name = name.replace("conv3.batch_norm", "bn3") 
        
        name = name.replace("neck", "encoder")
        name = name.replace("lateral", "lateral_conv")
        name = name.replace("lateral_conv.norm", "lateral_norm")
        name = name.replace("fpn", "fpn_conv")
        name = name.replace("fpn_conv.norm", "fpn_norm")
        if "dilated_encoder_blocks" in name:
            name = name.replace("conv1.weight", "conv1.0.weight")
            name = name.replace("conv1.bias", "conv1.0.bias")
            name = name.replace("conv1.norm", "conv1.1")
            name = name.replace("conv2.weight", "conv2.0.weight")
            name = name.replace("conv2.bias", "conv2.0.bias")
            name = name.replace("conv2.norm", "conv2.1")
            name = name.replace("conv3.weight", "conv3.0.weight")
            name = name.replace("conv3.bias", "conv3.0.bias")
            name = name.replace("conv3.norm", "conv3.1")
        
        if "kernel" in name:
            name = name.replace("kernel", "weight")
        if "head" in name:
            if "cls_score" not in name and "pred" not in name:
                name = _HEAD_WEIGHT_MAP[name]
            name = name.replace("head", "decoder")
        name = name.replace("predicted_class", "cls_score")
        name = name.replace("predicted_box", "bbox_pred")
        name = name.replace("predicted_objecteness", "object_pred")
        
        # print(name, weight.shape)
        tw = pretrained[name].numpy()
        if len(tw.shape) == 4:
            tw = np.transpose(tw, (2, 3, 1, 0))
        weight.assign(tw)


if __name__ == "__main__":
    import cv2
    import numpy as np
    from demo import coco_id_mapping, draw, random_color
    from configs.yolof_config import get_yolof_config
    from data.datasets.coco_dataset import COCODataset
    from core.metrics.mean_average_precision import mAP

    cfg = get_yolof_config()
    model = YOLOF(cfg, training=False)
    # model.detector(tf.random.uniform((1, 512, 512, 3)), training=False)
    name = "YOLOF_X_101_64x4d_C5_1x"
    _load_weight_from_torch(model.detector, "/home/bail/Data/data2/pretrained_weights/%s.pth" % name)
    cfg.save_to_yaml("./yamls/%s.yaml" % name)

    img = cv2.imread("/home/bail/Workspace/TRTNets/images/bus.jpg")
    img = cv2.resize(img, cfg.input_shape[:2])
    inp = img[:]
    inp = tf.convert_to_tensor(inp[None], tf.float32)
    inp = tf.concat([inp, inp], 0)
    
    outs = model(inp, training=False)
    
    num = outs["valid_detections"].numpy()[-1]
    boxes = outs["nmsed_boxes"].numpy()[-1]
    scores = outs["nmsed_scores"].numpy()[-1]
    classes = outs["nmsed_classes"].numpy()[-1]
    
    for i in range(num):
        # box = boxes[i] * np.array(list(cfg.input_shape[:2]) * 2)
        # box = boxes[i] * np.array([height, width, height, width])
        box = boxes[i]
        c = classes[i] + 1
        print(box, c, scores[i])
        img = draw(img, box, c, scores[i], coco_id_mapping, random_color(int(c)))
    
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    tf.saved_model.save(model.detector, "/home/bail/Data/data2/pretrained_weights/%s" % name)
    model.save_weights("/home/bail/Data/data2/pretrained_weights/%s.h5" % name)

