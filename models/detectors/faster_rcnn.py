import tensorflow as tf 
from ..builder import DETECTORS
from .two_stage import TwoStageDetector


@DETECTORS.register
class FasterRCNN(TwoStageDetector):
    def __init__(self, cfg, **kwargs):
        super(FasterRCNN, self).__init__(cfg, **kwargs)


_FPN_WEIGHTS_MAP = {
    "neck.lateral_convs.2.weight": "backbone.fpn_lateral2.weight",
    "neck.lateral_convs.2.bias": "backbone.fpn_lateral2.bias",
    "neck.fpn_convs.2.weight": "backbone.fpn_output2.weight",
    "neck.fpn_convs.2.bias" : "backbone.fpn_output2.bias",
    "neck.lateral_convs.3.weight": "backbone.fpn_lateral3.weight",
    "neck.lateral_convs.3.bias": "backbone.fpn_lateral3.bias",
    "neck.fpn_convs.3.weight": "backbone.fpn_output3.weight",
    "neck.fpn_convs.3.bias": "backbone.fpn_output3.bias",
    "neck.lateral_convs.4.weight": "backbone.fpn_lateral4.weight",
    "neck.lateral_convs.4.bias": "backbone.fpn_lateral4.bias",
    "neck.fpn_convs.4.weight": "backbone.fpn_output4.weight",
    "neck.fpn_convs.4.bias": "backbone.fpn_output4.bias",
    "neck.lateral_convs.5.weight": "backbone.fpn_lateral5.weight",
    "neck.lateral_convs.5.bias": "backbone.fpn_lateral5.bias",
    "neck.fpn_convs.5.weight": "backbone.fpn_output5.weight",
    "neck.fpn_convs.5.bias": "backbone.fpn_output5.bias",
}


def _load_weight_from_torch(model, torch_weights_path):
    import re
    import torch
    import pickle
    import numpy as np
    import torch.nn as nn

    # model.summary()
    with open(torch_weights_path, "rb") as f:
        pretrained = pickle.load(f, encoding="latin1")
    
    # pretrained = torch.load(f, map_location=torch.device("cpu"))["model"]
    # for k, v in pretrained["model"].items():
    #     if"tracked" not in k:
    #         print(k, v.shape)
    
    for weight in model.weights:
        name = weight.name
        # print(name, weight.shape)
        name = name.split(":")[0]
        name = name.replace("/", ".")

        if "stem" in name:
            name = "backbone.bottom_up." + name

        if "layer" in name:
            name = name.replace("layer1", "res2")
            name = name.replace("layer2", "res3")
            name = name.replace("layer3", "res4")
            name = name.replace("layer4", "res5")
            name = "backbone.bottom_up." + name
        
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
               
        if "kernel" in name:
            name = name.replace("kernel", "weight")
        
        if name in _FPN_WEIGHTS_MAP:
            name = _FPN_WEIGHTS_MAP[name]
        
        if "rpn_head" in name:
            name = "proposal_generator." + name
            name = name.replace("rpn_conv", "conv")
            name = name.replace("predicted_class", "objectness_logits")
            name = name.replace("predicted_box", "anchor_deltas")
        if "roi_heads" in name:
            name = name.replace("regressor", "box_predictor.bbox_pred")
            name = name.replace("classifier", "box_predictor.cls_score")

        # print(name, weight.shape)
        tw = pretrained["model"][name]
        if len(tw.shape) == 4:
            tw = np.transpose(tw, (2, 3, 1, 0))
        if len(tw.shape) == 2:
            tw = np.transpose(tw, (1, 0))
        weight.assign(tw)


if __name__ == "__main__":
    import cv2
    import numpy as np
    from demo import coco_id_mapping, draw, random_color, preprocess
    from configs.faster_rcnn_config import get_faster_rcnn_config
    from data.datasets.coco_dataset import COCODataset
    from core.metrics.mean_average_precision import mAP

    cfg = get_faster_rcnn_config()
    model = FasterRCNN(cfg, training=False)
    # model.detector(tf.random.uniform((1, 512, 512, 3)), training=False)
    name = "FasterRCNN_FPN_1x"
    _load_weight_from_torch(model.detector, "/home/bail/Downloads/model_final_b275ba.pkl")
    cfg.save_to_yaml("./yamls/%s.yaml" % name)

    img = cv2.imread("/home/bail/Workspace/TRTNets/images/bus.jpg")
    
    h, w = img.shape[0:2]
    raito = 800 / min(h, w)
    h, w = int(raito * h), int(w * raito)
    img = cv2.resize(img, (w, h))
    
    tmp = np.zeros([1088, 800, 3]) + np.array([[[103.5300, 116.2800, 123.6750]]])
    tmp[0:h, 0:w, :] = img 

    inp = tmp[:]
    inp = tf.convert_to_tensor(inp[None], tf.float32)
    # inp = tf.pad(img, [[0, 0], [0, 1088 - h], [0, 800 - w], [0, 0]], constant_values=(103, 116, 123))
    # inp = tf.concat([inp, inp], 0)
    
    proposals, outs = model(inp, training=False)
   
    # rois = proposals["rois"].numpy()[-1]
    # scores = proposals["scores"].numpy()[-1]
    # for i in range(len(rois)):
    #     box = rois[i]
    #     score = scores[i]
    #     if score > 0.99:
    #         img = draw(img, box, 1, score, coco_id_mapping, random_color(1))

    num = outs["valid_detections"].numpy()[-1]
    boxes = outs["nmsed_boxes"].numpy()[-1]
    scores = outs["nmsed_scores"].numpy()[-1]
    classes = outs["nmsed_classes"].numpy()[-1]
    
    for i in range(num):
        box = boxes[i]
        # box = boxes[i] * np.array([height, width, height, width])
        c = classes[i] + 1
        # print(box, c, scores[i])
        img = draw(img, box, c, scores[i], coco_id_mapping, random_color(int(c)))
    
    # cv2.imshow("img", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # tf.saved_model.save(model.detector, "/home/bail/Data/data2/pretrained_weights/%s" % name)
    # model.save_weights("/home/bail/Data/data2/pretrained_weights/%s.h5" % name)

