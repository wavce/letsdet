from configs import Config


def get_yolov5_config(num_classes=80, depth_multiple=.33, width_multiple=.50, name="yolov5s"):
    h = Config()
    
    h.detector = "YOLOv5"
    h.dtype = "float16"
    h.num_classes = num_classes
    h.depth_multiple = depth_multiple  # 0.33 0.67 1.0 1.33
    h.width_multiple = width_multiple  # 0.50 0.75 1.0 1.25
    h.model = [ 
        #        [from, number, module, args]
        # backbone
        [-1, 1, "Focus", dict(filters=64, kernel_size=3)],        # 0-P1/2
        [-1, 1, "Conv", dict(filters=128, kernel_size=3, strides=2)],
        [-1, 3, "BottleneckCSP", dict(filters=128)],  # 2-P3/8
        [-1, 1, "Conv", dict(filters=256, kernel_size=3, strides=2)],
        [-1, 9, "BottleneckCSP", dict(filters=256)],  # 4-P4/16
        [-1, 1, "Conv", dict(filters=512, kernel_size=3, strides=2)],
        [-1, 9, "BottleneckCSP", dict(filters=512)],  # 6-P5/32
        [-1, 1, "Conv", dict(filters=1024, kernel_size=3, strides=2)],
        [-1, 1, "SpatialPyramidPooling", dict(filters=1024, pool_sizes=[5, 9, 13])],
        [-1, 3, "BottleneckCSP", dict(filters=1024, shortcut=False)],  # 9 
        # head
        [-1, 1, "Conv", dict(filters=512, kernel_size=1, strides=1)],
        [-1, 1, "Upsample", dict(size=2, interpolation="nearest")],
        [[-1, 6], 1, "Concat", dict(axis=-1)],  # concat backbone P4
        [-1, 3, "BottleneckCSP", dict(filters=512, shortcut=False)],  # 13
        
        [-1, 1, "Conv", dict(filters=256, kernel_size=1, strides=1)],
        [-1, 1, "Upsample", dict(size=2, interpolation="nearest")],
        [[-1, 4], 1, "Concat", dict(axis=-1)],  # concat backbone P3
        [-1, 3, "BottleneckCSP", dict(filters=256, shortcut=False)],  # 17 (P3/8-small)
        
        [-1, 1, "Conv", dict(filters=256, kernel_size=3, strides=2)],
        [[-1, 14], 1, "Concat", dict(axis=-1)],  # concat head P4
        [-1, 3, "BottleneckCSP", dict(filters=512, shortcut=False)],  # 20 (P4/16-medium)
        
        [-1, 1, "Conv", dict(filters=512, kernel_size=3, strides=2)],
        [[-1, 10], 1, "Concat", dict(axis=-1)],  # concat head P5
        [-1, 3, "BottleneckCSP", dict(filters=1024, shortcut=False)],  # 23 (P5-large)
        
        [[17, 20, 23], 1, "Detect", dict(anchors="anchors", num_classes="num_classes")]
    ]
    h.min_level = 3
    h.max_level = 5
    h.strides = [8, 16, 32] 
    h.anchors = [[10, 13, 16, 30, 33, 23], 
                 [30, 61, 62, 45, 59, 119], 
                 [116, 90, 156, 198, 373, 326]]
    h.num_anchors = 3
    h.anchor_threshold = 4.  # 2.91 if finetuning else 4.0
    h.gr = 1.
        
    h.bbox_loss = dict(loss="CIoULoss", weight=1., reduction="none")  
    h.label_loss = dict(loss="BinaryCrossEntropy", weight=1., from_logits=True, reduction="none")  # .631 if finetuning else weight = 1.0
    h.conf_loss = dict(loss="BinaryCrossEntropy", weight=1., from_logits=True, reduction="none")   # 0.911 if finetuning else weight = 1.
    h.balance = [1., 1., 1.] # [4.0, 1.0, 0.4]   # if num_level == 3 else [4.0, 1.0, 0.4, 0.1]
    h.box_weight = 0.05  # 0.0296 if finetune else 0.05
    h.label_weight = .5  # 0.243 if finetune else 0.5
    h.conf_weight = 1.0   # 0.301 if finetune else 1.0
    
    h.weight_decay = 0.0005
    h.train=dict(dataset=dict(dataset="COCODataset",
                              batch_size=8,
                              dataset_dir="/data/bail/COCO",
                              training=True,
                              augmentations=[
                                  dict(FlipLeftToRight=dict(probability=0.5)),
                                  dict(RandomDistortColor=dict(probability=1.)),
                                #   dict(Resize=dict(size=(640, 640), strides=32, min_scale=.5, max_scale=2.)),
                                  dict(ResizeV2=dict(short_side=640, long_side=1024, strides=32, min_scale=1.0, max_scale=1.0))
                                  ],
                            #   mixup=dict(alpha=8.0, prob=0.5),
                              mosaic=dict(size=640, min_image_scale=0.25, prob=1.),
                              num_samples=118287),
                  pretrained_weights_path="/data/bail/pretrained_weights/darknet53-notop/darknet53.ckpt",
                  optimizer=dict(optimizer="SGD", momentum=0.937),
                  mixed_precision=dict(loss_scale=None),  # The loss scale in mixed precision training. If None, use dynamic.
                  gradient_clip_norm=.0,

                  scheduler=dict(train_epochs=480,
                                 #  learning_rate_scheduler=dict(scheduler="PiecewiseConstantDecay",
                                 #                               boundaries=[24, 32],
                                 #                               values=[0.012, 0.0012, 0.00012]),
                                 learning_rate_scheduler=dict(scheduler="CosineDecay", initial_learning_rate=0.012),
                                 warmup=dict(warmup_learning_rate=0.0012, steps=12000)),
                  checkpoint_dir="checkpoints/%s" % name,
                  summary_dir="logs/%s" % name,
                  log_every_n_steps=100,
                  save_ckpt_steps=10000)
    h.val=dict(dataset=dict(dataset="COCODataset", 
                            batch_size=8,  
                            dataset_dir="/data/bail/COCO", 
                            training=False, 
                            augmentations=[
                                dict(Resize=dict(size=(640, 640), strides=32, min_scale=1., max_scale=1.0))
                                # dict(ResizeV2=dict(short_side=800, long_side=1333, strides=64, min_scale=1.0, max_scale=1.0))
                            ]),
               samples=5000)
    h.test=dict(nms="NonMaxSuppressionWithQuality",
                pre_nms_size=5000,
                post_nms_size=100, 
                iou_threshold=0.6, 
                score_threshold=0.05,
                sigma=0.5,
                nms_type="nms")
    # h.test=dict(nms="CombinedNonMaxSuppression",
    #             pre_nms_size=5000,
    #             post_nms_size=100, 
    #             iou_threshold=0.6, 
    #             score_threshold=0.05)

    return h


if __name__ == "__main__":
    cfg = get_yolov5_config()
    cfg.save_to_yaml("./yolov5s.yaml")
