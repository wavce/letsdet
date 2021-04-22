from configs import Config


def default_detection_configs(phi,
                              num_classes=90,
                              min_level=3, 
                              max_level=7, 
                              fpn_filters=64,
                              neck_repeats=3,
                              head_repeats=3,
                              anchor_scale=4,
                              num_scales=3,
                              batch_size=4,
                              image_size=512,
                              fpn_name="BiFPN",
                              fpn_input_dims=[80, 192, 320],
                              fusion_type="weighted_sum"):
    h = Config()
    
    h.detector = "EfficientDetD%d" % phi
    h.dtype = "float32"
    h.num_classes = num_classes
    h.backbone = dict(backbone="EfficientNetB%d" % phi,
                      convolution="depthwise_conv2d",
                      dropblock=None,
                    #   dropblock=dict(keep_prob=None,
                    #                  block_size=None)
                      normalization=dict(normalization="batch_norm",
                                         momentum=0.99,
                                         epsilon=1e-3,
                                         axis=-1,
                                         trainable=False),
                      activation=dict(activation="swish"),
                      strides=[2, 1, 2, 2, 2, 1, 2, 1],
                      dilation_rates=[1, 1, 1, 1, 1, 1, 1, 1],
                      output_indices=[3, 4, 5],
                      frozen_stages=[-1, ])
    h.neck=dict(neck=fpn_name,
                input_size=image_size if isinstance(image_size, (list, tuple)) else [image_size, image_size],
                num_backbone_levels=3,
                feat_dims=fpn_filters,
                repeats=neck_repeats,
                convolution="separable_conv2d",
                normalization=dict(normalization="batch_norm", momentum=0.99, epsilon=1e-3, axis=-1, trainable=False),
                activation=dict(activation="swish"),
                min_level=3,
                max_level=7,
                input_dims=fpn_input_dims,
                pool_type=None, 
                apply_bn=True, 
                fusion_type=fusion_type)
    # anchors parameters
    strides = [2 ** l for l in range(min_level, max_level + 1)]
    scales=[
      [2 ** (i / num_scales) * s * anchor_scale 
      for i in range(num_scales)] for s in strides
    ]
    aspect_ratios = [1., 0.5, 2.]
    num_scales = len(scales[0]) * len(aspect_ratios)

    h.anchors = dict(aspect_ratios=aspect_ratios,
                     strides=strides,
                     scales=scales,
                     num_anchors=num_scales)
    h.head=dict(head="RetinaNetHead",
                convolution="separable_conv2d",
                normalization=dict(normalization="batch_norm", momentum=0.99, epsilon=1e-3, axis=-1, trainable=False),
                activation=dict(activation="swish"),
                feat_dims=fpn_filters,
                dropblock=None,
                repeats=head_repeats,
                min_level=min_level,
                max_level=max_level,
                use_sigmoid=True,
                prior=0.01,
                survival_prob=None,
                data_format="channels_last",
                bbox_encoder=dict(encoder="Box2Delta", weights=None),
                bbox_decoder=dict(decoder="Delta2Box", weights=None),
                assigner=dict(assigner="ATSSAssigner", topk=9),
                sampler=dict(sampler="PseudoSampler"),
                label_loss=dict(loss="FocalLoss", gamma=2.0, alpha=0.25, label_smoothing=0.01, weight=1., from_logits=True, reduction="sum"),
                bbox_loss=dict(loss="CIoULoss", weight=1., reduction="sum"))
    h.weight_decay = 4e-5
    h.train=dict(input_size=(image_size, image_size),
                 dataset=dict(dataset="COCODataset",
                              batch_size=batch_size,
                              dataset_dir="/data/bail/COCO",
                              training=True,
                              augmentations=[dict(FlipLeftToRight=dict(probability=0.5)),
                                             dict(RandomDistortColor=dict(probability=1.)),
                                             dict(Resize=dict(size=(image_size, image_size), strides=32, min_scale=0.5, max_scale=2.0)),
                                            #  dict(ResizeV2=dict(short_side=800, long_side=1333, strides=64, min_scale=1.0, max_scale=1.0))
                                            ],
                              num_samples=118287,
                              num_classes=num_classes),
                  pretrained_weights_path="/data/bail/pretrained_weights/resnet50/resnet50.ckpt",

                  optimizer=dict(optimizer="SGD", momentum=0.9),
                  mixed_precision=dict(loss_scale=None),  # The loss scale in mixed precision training. If None, use dynamic.
                  gradient_clip_norm=.0,

                  scheduler=dict(train_epochs=18,
                                #  learning_rate_scheduler=dict(scheduler="PiecewiseConstantDecay",
                                #                               boundaries=[8, 11],
                                #                               values=[0.02, 0.002, 0.0002]),
                                 learning_rate_scheduler=dict(scheduler="CosineDecay", initial_learning_rate=0.02),
                                 warmup=dict(warmup_learning_rate=0.0001, steps=800)),
                  checkpoint_dir="checkpoints/efficientdet-d%d" % phi,
                  summary_dir="logs/efficientdet-d%d" % phi,
                  log_every_n_steps=100,
                  save_ckpt_steps=5000)
    h.val=dict(dataset=dict(dataset="COCODataset", 
                            batch_size=batch_size,  
                            dataset_dir="/data/bail/COCO", 
                            training=False, 
                            augmentations=[
                                dict(Resize=dict(size=(image_size, image_size), strides=32, min_scale=1.0, max_scale=1.0))
                                # dict(ResizeV2=dict(short_side=800, long_side=1333, strides=64, min_scale=1.0, max_scale=1.0))
                            ]),
               input_size=(image_size, image_size),
               samples=5000)
    h.test=dict(nms="CombinedNonMaxSuppression",
                pre_nms_size=5000,   # select top_k high confident detections for nms 
                post_nms_size=100,
                iou_threshold=0.6,
                score_threshold=0.1,)

    return h


efficientdet_model_param_dict = {
    "EfficientDetD0": dict(phi=0, 
                           batch_size=32,
                           fpn_filters=64, 
                           neck_repeats=3, 
                           head_repeats=3, 
                           anchor_scale=4,
                           fpn_input_dims=[40, 112, 320],
                           image_size=512),
    "EfficientDetD1": dict(phi=1, 
                           batch_size=32,
                           fpn_filters=88, 
                           neck_repeats=4, 
                           head_repeats=3, 
                           anchor_scale=4,
                           fpn_input_dims=[80, 192, 320],
                           image_size=640),
    "EfficientDetD2": dict(phi=2, 
                            batch_size=4,
                            fpn_filters=112, 
                            neck_repeats=5, 
                            head_repeats=3, 
                            anchor_scale=4,
                            fpn_input_dims=[80, 192, 320],
                            image_size=768),
    "EfficientDetD3": dict(phi=3, 
                            batch_size=4,
                            fpn_filters=160, 
                            neck_repeats=6, 
                            head_repeats=4, 
                            anchor_scale=4.,
                            fpn_input_dims=[80, 192, 320],
                            image_size=896),
    "EfficientDetD4": dict(phi=4, 
                           batch_size=4,
                           fpn_filters=224, 
                           neck_repeats=7, 
                           head_repeats=4, 
                           anchor_scale=4.,
                           fpn_input_dims=[80, 192, 320],
                           image_size=1024),
    "EfficientDetD5": dict(phi=5, 
                           batch_size=4,
                           fpn_filters=288, 
                           neck_repeats=7, 
                           head_repeats=4, 
                           anchor_scale=4.,
                           fpn_input_dims=[80, 192, 320],
                           image_size=1280),
    "EfficientDetD6": dict(phi=6, 
                           batch_size=4,
                           fpn_filters=384, 
                           neck_repeats=8, 
                           head_repeats=5, 
                           anchor_scale=4.,
                           image_size=1280, 
                           fpn_input_dims=[80, 192, 320],
                           fusion_type="sum"),
    "EfficientDetD7": dict(phi=7, 
                           batch_size=4,
                           fpn_filters=384, 
                           neck_repeats=8, 
                           head_repeats=5, 
                           anchor_scale=5.,
                           image_size=1536, 
                           fpn_input_dims=[80, 192, 320],
                           fusion_type="sum"),
}


def get_efficientdet_config(model_name="EfficientDetD0", num_classes=90):
    return default_detection_configs(num_classes=num_classes, **efficientdet_model_param_dict[model_name])

