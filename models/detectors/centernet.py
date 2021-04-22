import math
import numpy as np
import tensorflow as tf
from utils import box_utils
from core import build_loss
from ..builder import build_head
from ..builder import build_backbone
from models.builder import DETECTORS
from .one_stage import OneStageDetector


@DETECTORS.register
class CenterNet(OneStageDetector):
    def __init__(self, cfg, **kwargs):
        super(CenterNet, self).__init__(cfg, **kwargs)


def _load_weight_from_torch(model, torch_weights_path="/home/bail/Downloads/ctdet_coco_hg.pth"):
    import torch
    import numpy as np

    loaded = torch.load(torch_weights_path)["state_dict"]
    # for k, v in loaded.items():
    #    if "tracked" not in k:
    #       print(k, v.shape)

    for weight in model.weights:
        
        name = weight.name.split(":")[0]
        name = name.replace("/", ".")
        # print(name, weight.shape.as_list())
        if "batch_norm" in name:
            name = name.replace("batch_norm", "bn")
        if "kernel" in name:
            name = name.replace("kernel", "weight")
        if "gamma" in name:
            name = name.replace("gamma", "weight")
        if "beta" in name:
            name = name.replace("beta", "bias")
        if "moving_mean" in name:
            name = name.replace("moving_mean", "running_mean")
        if "moving_variance" in name:
            name = name.replace("moving_variance", "running_var")
        if "conv2d.weight" in name:
            name = name.replace("conv2d.weight", "conv.weight")
        if "conv2d.bias" in name:
            name = name.replace("conv2d.bias", "conv.bias")
        if "skip.bn" in name:
            name = name.replace("skip.bn", "skip.1")
        if "skip.conv" in name:
            name = name.replace("skip.conv", "skip.0")
        if "cnvs_.0.bn" in name:
            name = name.replace("cnvs_.0.bn", "cnvs_.0.1")
        if "cnvs_.0.conv" in name:
            name = name.replace("cnvs_.0.conv", "cnvs_.0.0")
        if "inters_.0.bn" in name:
            name = name.replace("inters_.0.bn", "inters_.0.1")
        if "inters_.0.conv" in name:
            name = name.replace("inters_.0.conv", "inters_.0.0")
        if "conv1.conv" in name:
            name = name.replace("conv1.conv", "conv1")
        if "conv1.bn" in name:
            name = name.replace("conv1.bn", "bn1")
        if "conv2.conv" in name:
            name = name.replace("conv2.conv", "conv2")
        if "conv2.bn" in name:
            name = name.replace("conv2.bn", "bn2")

        name = "module." + name
        if "module.head" in name:
            name = name.replace("module.head", "module")
        if "predicted_reg" in name:
            name = name.replace("predicted_reg", "")
        if "predicted_hm" in name:
            name = name.replace("predicted_hm", "")
        if "predicted_wh" in name:
            name = name.replace("predicted_wh", "")

        # print(name)
        tw = loaded[name].numpy()
        if len(tw.shape) == 4:
            tw = np.transpose(tw, (2, 3, 1, 0))
      
        if len(tw.shape) == 2:
            tw = np.transpose(tw, (1, 0))
        weight.assign(tw)


def _load_weight_from_torch2(model, torch_weights_path="/home/bail/Downloads/ctdet_coco_dla_1x.pth"):
    import re
    import torch
    import numpy as np
    import torch.nn as nn

    # model.summary()
    t_dla = torch.load(torch_weights_path)["state_dict"]
   #  for k, v in t_dla.items():
   #     rint(k, v.shape)
    
    for weight in model.weights:
        name = weight.name
        # print(name, weight.shape)
        name = name.split(":")[0]
        name = name.replace("/", ".")
        if "base_layer.conv2d.kernel" in name:
            name = name.replace("conv2d.kernel", "0.weight")
        if "base_layer.batch_norm" in name:
            name = name.replace("batch_norm", "1")
        if "level0.0.conv2d.kernel" in name:
            name = name.replace("conv2d.kernel", "weight")
        if "level0.0.batch_norm" in name:
            name = name.replace("0.batch_norm", "1")
        if "level1.0.conv2d.kernel" in name:
            name = name.replace("conv2d.kernel", "weight")
        if "level1.0.batch_norm" in name:
            name = name.replace("0.batch_norm", "1")
        for i in range(3):
            if "conv%d.batch_norm" % (i + 1) in name:
                name = name.replace("conv%d.batch_norm" % (i + 1), "bn%d" % (i + 1))
        if "project.conv2d" in name:
            name = name.replace("project.conv2d.kernel", "project.0.weight")
        if "project.batch_norm" in name:
            name = name.replace("project.batch_norm", "project.1")
        if "root.conv.batch_norm" in name:
            name = name.replace("root.conv.batch_norm", "root.bn")
        if "conv2d.kernel" in name:
            name = name.replace("conv2d.kernel", "weight")
        if "gamma" in name:
            name = name.replace("gamma", "weight")
        if "beta" in name:
            name = name.replace("beta", "bias")
        if "moving_mean" in name:
            name = name.replace("moving_mean", "running_mean")
        if "moving_variance" in name:
            name = name.replace("moving_variance", "running_var")
        if "kernel" in name:
            name = name.replace("kernel", "weight")
        name = re.sub("_\d+\d*", "", name)
        if "dla_up" not in name:
            name = "module.base." + name

        tw = t_dla[name].numpy()
        if len(tw.shape) == 4:
            tw = np.transpose(tw, (2, 3, 1, 0))
        weight.assign(tw)


if __name__ == "__main__":
    import cv2
    import numpy as np
    from demo import coco_id_mapping, draw, random_color
    from configs.centernet_config import get_centernet_config
    from data.datasets.coco_dataset import COCODataset
    from core.metrics.mean_average_precision import mAP

    cfg = get_centernet_config()
    model = CenterNet(cfg, training=False)
    model.head.summary()
    # model.detector(tf.random.uniform((1, 512, 512, 3)), training=False)
    torch_weight_name = "ctdet_coco_hg.pth"
    _load_weight_from_torch(model.detector, "/home/bail/Data/data2/pretrained_weights/%s" % torch_weight_name)

    img1 = cv2.imread("/home/bail/Workspace/TRTNets/images/bus.jpg")
    img1 = cv2.resize(img1, (512, 512))
    inp1 = img1[:, :, ::-1]
    img2 = cv2.imread("/home/bail/Workspace/TensorRT/TRTNets/images/zidane.jpg")
    img2 = cv2.resize(img2, (512, 512))
    inp2 = img2[:, :, ::-1]

    inp1 = tf.convert_to_tensor(inp1[None], tf.float32)
    inp2 = tf.convert_to_tensor(inp2[None], tf.float32)
    # inp = tf.concat([inp2, inp1], 0)
    outs = model(inp1, training=False)
    # hm = tf.reduce_max(tf.nn.sigmoid(outs["hm"][1]), -1, keepdims=True).numpy()[0] * 255
    # cv2.imshow("heatmap", hm.astype(np.uint8))
    # cv2.waitKey(0)
    
    num = outs["valid_detections"].numpy()[-1]
    boxes = outs["nmsed_boxes"].numpy()[-1]
    scores = outs["nmsed_scores"].numpy()[-1]
    classes = outs["nmsed_classes"].numpy()[-1]
    
    for i in range(num):
        box = boxes[i]
        c = classes[i] + 1
        img1 = draw(img1, box, c, scores[i], coco_id_mapping, random_color(int(c)))
    
    cv2.imshow("img", img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    tf.saved_model.save(model.detector, "/home/bail/Data/data2/pretrained_weights/%s" % torch_weight_name)
    model.save_weights("/home/bail/Data/data2/pretrained_weights/%s.h5" % torch_weight_name)
