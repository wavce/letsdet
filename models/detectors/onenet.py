import numpy as np
import tensorflow as tf 
from . import Detector
from ..builder import DETECTORS
from ..builder import build_head
from ..builder import build_neck
from ..builder import build_backbone


@DETECTORS.register
class OneNet(Detector):
    def __init__(self, cfg, training=True, **kwargs):
        super(OneNet, self).__init__(cfg=cfg, **kwargs)

        self.data_format = cfg.data_format
       
        inputs = tf.keras.Input(shape=cfg.input_shape)
        self.backbone = build_backbone(input_tensor=inputs, **cfg.backbone.as_dict())
        x = self.backbone(inputs)
        input_shapes = [i.shape.as_list()[1:] for i in x]
        self.neck = build_neck(input_shapes=input_shapes, name="neck", **cfg.neck.as_dict())
        x = self.neck(x)
        self.head = build_head(cfg.head.head, 
                               cfg=cfg.head, 
                               test_cfg=cfg.test, 
                               num_classes=cfg.num_classes, 
                               is_training=training,
                               name="head")
        x = self.head(x)
       
        self.detector = tf.keras.Model(inputs=inputs, outputs=x)
    
    def save_weights(self, name):
        self.detector.save_weights(name)
    
    def compute_losses(self, predictions, image_info):
        return self.head.comput_losses(predictions, image_info)
    
    @tf.function
    def __call__(self, inputs, training):
        x = self.detector(inputs, training=training)
        return x


def _load_weight_from_torch(model, torch_weights_path="/home/bail/Downloads/onenet_r18nodcn.pth"):
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
        
        if "neck" in name:
            name = name.replace("neck", "head.deconv")
        if "norm" in name and "head" in name:
            name = name.replace("conv.norm", "bn")
        if "kernel" in name:
            name = name.replace("kernel", "weight")
        name = name.replace("predicted_class", "cls_score")
        name = name.replace("predicted_box", "ltrb_pred")

        tw = pretrained[name].numpy()
        if len(tw.shape) == 4:
            tw = np.transpose(tw, (2, 3, 1, 0))
        weight.assign(tw)


if __name__ == "__main__":
    import cv2
    import numpy as np
    from demo import coco_id_mapping, draw, random_color
    from configs.onenet_config import get_onenet_config
    from data.datasets.coco_dataset import COCODataset
    from core.metrics.mean_average_precision import mAP

    cfg = get_onenet_config()
    model = OneNet(cfg, training=False)
    # model.detector(tf.random.uniform((1, 512, 512, 3)), training=False)
    _load_weight_from_torch(model.detector, "/home/bail/Data/data2/pretrained_weights/onenet_r18nodcn.pth")

    img = cv2.imread("/home/bail/Workspace/TRTNets/images/bus.jpg")
    img = cv2.resize(img, (512, 512))
    inp = img[:, :, ::-1]
    inp = tf.convert_to_tensor(inp[None], tf.float32)
    inp = tf.concat([inp, inp], 0)
    
    outs = model(inp, training=False)
    #    hm = tf.reduce_max(tf.nn.sigmoid(outs["hm"][1]), -1, keepdims=True).numpy()[0]
    #    cv2.imshow("heatmap", hm)
    #    cv2.waitKey(0)
    
    num = outs["valid_detections"].numpy()[-1]
    boxes = outs["nmsed_boxes"].numpy()[-1]
    scores = outs["nmsed_scores"].numpy()[-1]
    classes = outs["nmsed_classes"].numpy()[-1]
    
    for i in range(num):
        box = boxes[i] 
        # box = boxes[i] * np.array([height, width, height, width])
        c = classes[i] + 1
        print(box, c)
        img = draw(img, box, c, scores[i], coco_id_mapping, random_color(int(c)))
    
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    tf.saved_model.save(model.detector, "/home/bail/Data/data2/pretrained_weights/onenet_r18")
    model.save_weights("/home/bail/Data/data2/pretrained_weights/onenet_r18.h5")

