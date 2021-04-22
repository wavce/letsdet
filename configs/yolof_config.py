from configs import Config


def get_yolof_config(num_classes=80):
    h = Config()
    
    data_format = "channels_last"
    input_size = (1024, 1024)
    h.detector = "YOLOF"
    h.dtype = "float16"
    h.data_format = data_format
    h.input_shape = (input_size[0], input_size[1], 3)
    h.num_classes = num_classes
    h.backbone = dict(backbone="ResNeXt101_64X4D",
                      dropblock=None, 
                      normalization=dict(normalization="batch_norm", momentum=0.997, epsilon=1e-4, trainable=False),
                      activation=dict(activation="relu"),
                      strides=[2, 2, 2, 2, 2],
                      dilation_rates=[1, 1, 1, 1, 1],
                      output_indices=[5],
                      frozen_stages=[1, ])
    h.neck=dict(neck="DilatedEncoder", 
                filters=512,
                midfilters=128,
                dilation_rates=[2, 4, 6, 8],  # dilation not in stage5
                # dilation_rates=[4, 8, 12, 16],
                normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-3, axis=-1, trainable=True),
                activation=dict(activation="relu"),
                kernel_initializer="he_normal",
                data_format=data_format)
    h.anchors=dict(generator="AnchorGenerator",
                   aspect_ratios=[1.],
                   scales=[32, 64, 128, 256, 512],
                   strides=32, 
                #    scales=[16, 32, 64, 128, 256, 512],
                #    strides=16, 
                   num_anchors=5)
    h.head=dict(head="YOLOFHead",
                normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-3, axis=-1, trainable=True),
                activation=dict(activation="relu"),
                feat_dims=512,
                cls_num_convs=2,
                reg_num_convs=4,
                kernel_initializer="he_normal",
                use_sigmoid=True,
                prior=0.01,
                bbox_decoder=dict(decoder="Delta2Box", weights=[1., 1., 1., 1.]),
                bbox_encoder=dict(encoder="Box2Delta", weights=[1., 1., 1., 1.]),
                assigner=dict(assigner="UniformAssigner", match_times=8, pos_ignore_thresh=0.7, neg_ignore_thresh=0.15),
                sampler=dict(sampler="PseudoSampler"),
                label_loss=dict(loss="FocalLoss", alpha=0.25, gamma=2.0, weight=1., from_logits=True, reduction="sum"),
                bbox_loss=dict(loss="GIoULoss", weight=2., reduction="sum"))
   
    h.weight_decay = 1e-4
    h.excluding_weight_names = ["predicted_box", "predicted_class"]
    h.train=dict(dataset=dict(dataset="COCODataset",
                              batch_size=4,
                              dataset_dir="/data/bail/COCO",
                              training=True,
                              augmentations=[
                                  dict(augmentation="FlipLeftToRight", probability=0.5),
                                  dict(augmentation="RandomDistortColor"),
                                  dict(augmentation="Resize", img_scale=[(1333, 800)], keep_ratio=True),
                                  dict(augmentation="Pad", size_divisor=32)
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
                  checkpoint_dir="checkpoints/yolof",
                  summary_dir="logs/yolof",
                  log_every_n_steps=100,
                  save_ckpt_steps=5000)
    h.val=dict(dataset=dict(dataset="COCODataset", 
                            batch_size=4,  
                            dataset_dir="/data/bail/COCO", 
                            training=False, 
                            augmentations=[
                                dict(augmentation="Resize", img_scale=[(1333, 800)], keep_ratio=True),
                                dict(augmentation="Pad", size_divisor=32),
                            ]),
               samples=5000)
    h.test=dict(nms="CombinedNonMaxSuppression",
                pre_nms_size=2000,
                post_nms_size=100, 
                iou_threshold=0.5, 
                score_threshold=0.35)

    return h
