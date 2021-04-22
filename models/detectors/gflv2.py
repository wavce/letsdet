import tensorflow as tf
from .gfl import GFL
from models.builder import DETECTORS


@DETECTORS.register
class GFLV2(GFL):
    def __init__(self, cfg, **kwargs):
        super(GFLV2, self).__init__(cfg, **kwargs)
    

  
_FPN_NAME_MAP = {
    "neck.lateral_convs.3.conv2d.weight": "neck.lateral_convs.0.conv.weight",
    "neck.lateral_convs.3.conv2d.bias": "neck.lateral_convs.0.conv.bias",
    "neck.fpn_convs.3.conv2d.weight": "neck.fpn_convs.0.conv.weight",
    "neck.fpn_convs.3.conv2d.bias": "neck.fpn_convs.0.conv.bias",
    "neck.lateral_convs.4.conv2d.weight": "neck.lateral_convs.1.conv.weight",
    "neck.lateral_convs.4.conv2d.bias": "neck.lateral_convs.1.conv.bias",
    "neck.fpn_convs.4.conv2d.weight": "neck.fpn_convs.1.conv.weight",
    "neck.fpn_convs.4.conv2d.bias": "neck.fpn_convs.1.conv.bias",
    "neck.lateral_convs.5.conv2d.weight": "neck.lateral_convs.2.conv.weight",
    "neck.lateral_convs.5.conv2d.bias": "neck.lateral_convs.2.conv.bias",
    "neck.fpn_convs.5.conv2d.weight": "neck.fpn_convs.2.conv.weight",
    "neck.fpn_convs.5.conv2d.bias": "neck.fpn_convs.2.conv.bias",
    "neck.fpn_convs.6.conv2d.weight": "neck.fpn_convs.3.conv.weight",
    "neck.fpn_convs.6.conv2d.bias": "neck.fpn_convs.3.conv.bias",
    "neck.fpn_convs.7.conv2d.weight": "neck.fpn_convs.4.conv.weight",
    "neck.fpn_convs.7.conv2d.bias": "neck.fpn_convs.4.conv.bias",
}


def _get_weights_from_pretrained(model, pretrained_weights_path):
    import torch
    import numpy as np

    pretrained = torch.load(pretrained_weights_path, map_location="cpu")["state_dict"]
    # for k, v in pretrained.items():
    #     if "tracked" not in k:
    #         print(k, v.numpy().shape)
    
    for w in model.weights:
        name = w.name.split(":")[0]
        name = name.replace("/", ".")
        
        if "shortcut.conv2d.kernel" in name:
            name = name.replace("shortcut.conv2d.kernel", "downsample.0.weight")
        if "shortcut.batch_norm" in name:
            name = name.replace("shortcut.batch_norm", "downsample.1")
        if "conv2d.kernel" in name and "neck" not in name and "head" not in name:
            name = name.replace("conv2d.kernel", "weight")
        if "conv2d.bias" in name and "neck" not in name and "head" not in name:
            name = name.replace("conv2d.bias", "bias")
        if "kernel" in name and "head" not in name:
            name = name.replace("kernel", "weight")
        if "gamma" in name and "head" not in name:
            name = name.replace("gamma", "weight")
        if "beta" in name and "head" not in name:
            name = name.replace("beta", "bias")
        if "moving_mean" in name:
            name = name.replace("moving_mean", "running_mean")
        if "moving_variance" in name:
            name = name.replace("moving_variance", "running_var")
        if "stem" in name: 
            name = name.replace("stem", "backbone")
        if "layer" in name:
            name = "backbone." + name
        if "conv1.batch_norm" in name:
            name = name.replace("conv1.batch_norm", "bn1")
        if "conv2.batch_norm" in name:
            name = name.replace("conv2.batch_norm", "bn2")
        if "conv3.batch_norm" in name:
            name = name.replace("conv3.batch_norm", "bn3") 
        if name in _FPN_NAME_MAP:
            name = _FPN_NAME_MAP[name]

        if "kernel" in name and "head" in name:
            name = name.replace("conv2d.kernel", "conv.weight")
        if "gamma" in name and "head" in name:
            name = name.replace("group_norm.gamma", "gn.weight")
        if "beta" in name and "head" in name:
            name = name.replace("group_norm.beta", "gn.bias")
    
        if "head" in name:
            name = name.replace("head", "bbox_head")
        if "box_net" in name:
            name = name.replace("box_net", "reg_convs")
        if "cls_net" in name:
            name = name.replace("cls_net", "cls_convs")
        
        if "class" in name:
            name = name.replace("predicted_class.kernel", "gfl_cls.weight")
            name = name.replace("predicted_class.bias", "gfl_cls.bias")
        if "box" in name and "box_net" not in name:
            name = name.replace("predicted_box.kernel", "gfl_reg.weight")
            name = name.replace("predicted_box.bias", "gfl_reg.bias")
        
        name = name.replace("kernel", "weight")

        # print(name, w.shape.as_list())
        pw = pretrained[name]
        if len(pw.shape) == 4:
            pw = np.transpose(pw, [2, 3, 1, 0])
        if len(pw.shape) == 2:
            pw = np.transpose(pw, [1, 0])
        if "scale" in name:
            pw = pw
            
        w.assign(pw)


if __name__ == "__main__":
    import cv2
    import numpy as np
    from demo import coco_id_mapping, draw, random_color, preprocess
    from configs.gflv2_config import get_gflv2_config
    from data.datasets.coco_dataset import COCODataset
    from core.metrics.mean_average_precision import mAP

    cfg = get_gflv2_config()
    model = GFLV2(cfg, training=False)
    # model.backbone.summary()
    # model.neck.summary()
    # model.detector.summary()
    # model.detector(tf.random.uniform((1, 512, 512, 3)), training=False)
    torch_weight_name = "gflv2_r101_fpn_ms2x"
    cfg.save_to_yaml("./yamls/%s.yaml" % torch_weight_name)
    _get_weights_from_pretrained(model.detector, "/home/bail/Data/data2/pretrained_weights/%s.pth" % torch_weight_name)

    img = cv2.imread("/home/bail/Workspace/TRTNets/images/bus.jpg")
    img, _, _, _ = preprocess(img, 1024)
    img = img[:, :, ::-1]
    inp = tf.convert_to_tensor(img[None], tf.float32)
    inp = tf.concat([inp, inp], 0)
    
    outs = model(inp, training=False)
    
    num = outs["valid_detections"].numpy()[1]
    boxes = outs["nmsed_boxes"].numpy()[1]
    scores = outs["nmsed_scores"].numpy()[1]
    classes = outs["nmsed_classes"].numpy()[1]
    
    for i in range(num):
        box = boxes[i]
        c = classes[i] + 1
        print(box, c)
        
        img = draw(img, box, c, scores[i], coco_id_mapping, random_color(int(c)))
    
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    tf.saved_model.save(model.detector, "/home/bail/Data/data2/pretrained_weights/%s" % torch_weight_name)
    model.save_weights("/home/bail/Data/data2/pretrained_weights/%s.h5" % torch_weight_name)

