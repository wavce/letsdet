from configs import Config


def get_retinanet_config(num_classes=80):
    h = Config()
    
    input_size = (1024, 1024)
    h.detector = "GFL"
    h.dtype = "float16"
    h.data_format = "channels_last"
    h.input_shape = (input_size[0], input_size[1], 3)
    h.num_classes = num_classes
    h.backbone = dict(backbone="ResNeXt101_64X4D",
                      dropblock=None, 
                      normalization=dict(normalization="batch_norm", momentum=0.997, epsilon=1e-4, trainable=False),
                      activation=dict(activation="relu"),
                      strides=[2, 2, 2, 2, 2],
                      dilation_rates=[1, 1, 1, 1, 1],
                      output_indices=[3, 4, 5],
                      frozen_stages=[1, ])
    h.neck=dict(neck="FPN", 
                feat_dims=256,
                min_level=3, 
                max_level=5,
                num_output_levels=5, 
                add_extra_convs=True,
                add_extra_convs_on_c5=True,
                relu_before_extra_convs=False)
    h.anchors=dict(generator="AnchorGeneratorV2",
                   aspect_ratios=[0.5, 1., 2.], 
                   octave_base_scale=4,
                   scales_per_octave=3,
                   strides=[8, 16, 32, 64, 128], 
                   num_anchors=9)
    h.head=dict(head="RetinaNetHead",
                normalization=None,
                activation=dict(activation="relu"),
                feat_dims=256,
                dropblock=None,
                num_classes=num_classes,
                repeats=4,
                min_level=3,
                max_level=7,
                use_sigmoid=True,
                prior=0.01,
                bbox_decoder = dict(decoder="Delta2Box", weights=[1., 1., 1., 1.]),
                bbox_encoder = dict(encoder="Box2Delta", weights=[1., 1., 1., 1.]),
                assigner = dict(assigner="MaxIoUAssigner", pos_iou_thresh=0.5, neg_iou_thresh=0.4),
                sampler = dict(sampler="PseudoSampler"),
                label_loss = dict(loss="FocalLoss", alpha=0.25, gamma=2., weight=1., from_logits=True, reduction="sum"),
                bbox_loss = dict(loss="SmoothL1Loss", weight=1., reduction="sum"))
   
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
                                  dict(augmentation="Pad", size_divisor=32),
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
                  checkpoint_dir="checkpoints/retinanet",
                  summary_dir="logs/retinanet",
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
                pre_nms_size=5000,
                post_nms_size=100, 
                iou_threshold=0.5, 
                score_threshold=0.05)

    return h
