from configs import Config


def get_fcos_config(num_classes=80):
    h = Config()
    
    input_size = 1024
    h.detector = "FCOS"
    h.dtype = "float32"
    h.data_format = "channels_last"
    h.input_shape = (input_size, input_size, 3)
    h.num_classes = num_classes
    h.backbone = dict(backbone="CaffeResNet50",
                      dropblock=None, 
                      normalization=dict(normalization="batch_norm", momentum=0.997, epsilon=1e-4, trainable=False),
                      activation=dict(activation="relu"),
                      strides=[2, 2, 2, 2, 2],
                      dilation_rates=[1, 1, 1, 1, 1],
                      output_indices=[3, 4, 5],
                      frozen_stages=[-1, ])
    h.neck=dict(neck="FPN", 
                feat_dims=256,
                min_level=3, 
                max_level=5,
                num_output_levels=5, 
                add_extra_convs=True,
                relu_before_extra_convs=True)
    h.head=dict(head="FCOSHead",
                normalization=dict(normalization="group_norm", groups=32, trainable=True),
                activation=dict(activation="relu"),
                feat_dims=256,
                dropblock=None,
                num_classes=num_classes,
                centerness_on_box=True,
                repeats=4,
                min_level=3,
                max_level=7,
                object_sizes_of_interest=[[0, 64], [64, 128], [128, 256], [256, 512], [512, 1e10]],
                use_sigmoid=True,
                prior=0.01,
                normalize_box=True,
                assigner=dict(assigner="FCOSAssigner", sampling_radius=1.5),
                sampler=dict(sampler="PseudoSampler"),
                label_loss=dict(loss="FocalLoss", gamma=2.0, alpha=0.25, label_smoothing=0.01, weight=1., from_logits=True, reduction="sum"),
                bbox_loss=dict(loss="IoULoss", weight=1., reduction="sum"),
                centerness_loss=dict(loss="BinaryCrossEntropy", weight=1.0, from_logits=True, reduction="sum"))
    h.weight_decay = 0.0001
    h.excluding_weight_names = []
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
                  checkpoint_dir="checkpoints/fcos",
                  summary_dir="logs/fcos",
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
                iou_threshold=0.6, 
                score_threshold=0.3,
                nms_type="nms")

    return h



