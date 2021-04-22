from configs import Config


def get_onenet_config(num_classes=80):
    h = Config()
    
    input_size = (512, 512)
    h.detector = "OneNet"
    h.dtype = "float16"
    h.data_format = "channels_last"
    h.input_shape = (input_size[0], input_size[1], 3)
    h.num_classes = num_classes
    h.backbone = dict(backbone="ResNet18",
                      dropblock=None, 
                      normalization=dict(normalization="batch_norm", momentum=0.997, epsilon=1e-4, trainable=False),
                      activation=dict(activation="relu"),
                      strides=[2, 2, 2, 2, 2],
                      dilation_rates=[1, 1, 1, 1, 1],
                      output_indices=[2, 3, 4, 5],
                      frozen_stages=[1, ])
    
    h.neck=dict(neck="CenterNetDeconv", 
                normalization=dict(normalization="batch_norm", momentum=0.997, epsilon=1e-4, trainable=False),
                activation=dict(activation="relu"))
    h.head=dict(head="OneNetHead",
                activation=dict(activation="relu"),
                feat_dims=64,
                dropblock=None,
                num_classes=num_classes,
                strides=4,
                prior=0.01,
                use_sigmoid=True,
                assigner = dict(assigner="MinCostAssigner", class_weight=2., l1_weight=2., iou_weight=5., iou_type="giou", alpha=0.25, gamma=2.),
                label_loss = dict(loss="FocalLoss", alpha=0.25, gamma=2., reduction="sum"),
                bbox_loss = dict(loss="RegL1Loss", weight=1., reduction="sum"))
  
    h.weight_decay = 1e-4
    h.excluding_weight_names = ["predicted_box", "predicted_class"]
    h.train=dict(dataset=dict(dataset="COCODataset",
                              batch_size=4,
                              dataset_dir="/data/bail/COCO",
                              training=True,
                              augmentations=[
                                  dict(augmentation="FlipLeftToRight", probability=0.5),
                                  dict(augmentation="RandomDistortColor"),
                                  dict(augmentation="Resize", img_scale=(0.2, 2), multiscale_mode="range", keep_ratio=True),
                                  dict(augmentation="RandCropOrPad", size=(input_size, input_size), clip_box_base_center=False),
                              ],
                              num_samples=118287),
                  pretrained_weights_path="/data/bail/pretrained_weights/resnet50/resnet50.ckpt",

                  optimizer=dict(optimizer="SGD", momentum=0.9),
                  mixed_precision=dict(loss_scale=None),  # The loss scale in mixed precision training. If None, use dynamic.
                  gradient_clip_norm=10.0,

                  scheduler=dict(train_epochs=24,
                                 learning_rate_scheduler=dict(scheduler="PiecewiseConstantDecay",
                                                              boundaries=[16, 22],
                                                              values=[0.02, 0.002, 0.0002]),
                                 warmup=dict(warmup_learning_rate=0.001, steps=800)),
                  checkpoint_dir="checkpoints/onenet",
                  summary_dir="logs/onenet",
                  log_every_n_steps=100,
                  save_ckpt_steps=5000)
    h.val=dict(dataset=dict(dataset="COCODataset", 
                            batch_size=4,  
                            dataset_dir="/data/bail/COCO", 
                            training=False, 
                            augmentations=[
                                dict(augmentation="Resize", img_scale=[(1333, input_size)], keep_ratio=True),
                                dict(augmentation="Pad", size_divisor=32)
                            ]),
               samples=5000)
    h.test=dict(topk=100, score_threshold=0.3)

    return h
