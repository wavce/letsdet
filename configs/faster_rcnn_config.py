from configs import Config


def get_faster_rcnn_config(num_classes=80):
    h = Config()
    
    h.detector = "FasterRCNN"
    h.dtype = "float16"
    h.data_format = "channels_last"
    h.num_classes = num_classes
    h.backbone = dict(backbone="CaffeResNet50",
                      dropblock=None, 
                      normalization=dict(normalization="batch_norm", momentum=0.997, epsilon=1e-4, trainable=True),
                      activation=dict(activation="relu"),
                      strides=[2, 2, 2, 2, 2],
                      dilation_rates=[1, 1, 1, 1, 1],
                      output_indices=[2, 3, 4, 5],
                      frozen_stages=[-1, ])
    h.neck=dict(neck="FPN", 
                feat_dims=256, 
                min_level=2, 
                max_level=5, 
                num_output_levels=5,
                add_extra_convs=False)
    h.anchors=dict(generator="AnchorGenerator",
                   scales=[[32], [64], [128], [256], [512]], 
                   aspect_ratios=[0.5, 1., 2.0], 
                   strides=[4, 8, 16, 32, 64], 
                   num_anchors=3)
    h.rpn_head=dict(head="RPNHead",
                    normalization=None,
                    activation=dict(activation="relu"),
                    feat_dims=256,
                    dropblock=None,
                    num_classes=1,
                    min_level=2,
                    max_level=6,
                    use_sigmoid=True,
                    train_proposal=dict(pre_nms_size=2000, post_nms_size=1000, iou_threshold=0.7, min_box_size=0.), 
                    test_proposal=dict(pre_nms_size=1000, post_nms_size=1000, iou_threshold=0.7, min_box_size=0.1),
                    bbox_encoder=dict(encoder="Box2Delta", weights=None),
                    bbox_decoder=dict(decoder="Delta2Box", weights=None),
                    assigner=dict(assigner="MaxIoUAssigner", pos_iou_thresh=0.7, neg_iou_thresh=0.3, min_pos_iou=0.3),
                    sampler=dict(sampler="RandomSampler", num_proposals=256, pos_fraction=0.5, add_gt_as_proposals=False),
                    label_loss=dict(loss="CrossEntropy", label_smoothing=0.0, weight=1., from_logits=True, reduction="none"),
                    bbox_loss=dict(loss="SmoothL1Loss", delta=1. / 9., weight=1., reduction="none"))
    h.roi_head=dict(head="StandardRoIHead",
                    bbox_head = dict(roi_pooling=dict(roi_pooling="MultiLevelAlignedRoIPooling", pooled_size=7, feat_dims=256),
                                     normalization=None,
                                     activation=dict(activation="relu"),
                                     dropblock=None,
                                     num_convs=0,
                                     conv_dims=256,
                                     num_fc=2,
                                     fc_dims=1024,
                                     feat_dims=256),
                    min_level=2,
                    max_level=5,
                    class_agnostic=False,
                    use_sigmoid=False,
                    bbox_encoder=dict(encoder="Box2Delta", weights=[10., 10., 5., 5.]),
                    bbox_decoder=dict(decoder="Delta2Box", weights=[10., 10., 5., 5.]),
                    assigner=dict(assigner="MaxIoUAssigner", pos_iou_thresh=0.5, neg_iou_thresh=0.5, min_pos_iou=0.5),
                    sampler=dict(sampler="RandomSampler", num_proposals=512, pos_fraction=0.25, add_gt_as_proposals=True),
                    label_loss=dict(loss="CrossEntropy", label_smoothing=0.0, weight=1., from_logits=True, reduction="none"),
                    bbox_loss=dict(loss="SmoothL1Loss", delta=1., weight=1., reduction="none"),
                    reg_class_agnostic=True)
    h.weight_decay = 1e-4
    h.train=dict(dataset=dict(dataset="COCODataset",
                              batch_size=2,
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
                  gradient_clip_norm=10.0,
                  scheduler=dict(train_epochs=12,
                                 learning_rate_scheduler=dict(scheduler="PiecewiseConstantDecay",
                                                              boundaries=[8, 11],
                                                              values=[0.01, 0.001, 0.0001]),
                                 warmup=dict(warmup_learning_rate=0.00001, steps=2000)),
                  checkpoint_dir="checkpoints/faster_rcnn",
                  summary_dir="logs/faster_rcnn",
                  log_every_n_steps=100,
                  save_ckpt_steps=5000)
    h.val=dict(dataset=dict(dataset="COCODataset", 
                            batch_size=1,  
                            dataset_dir="/data/bail/COCO", 
                            training=False, 
                            augmentations=[
                                dict(augmentation="Resize", img_scale=[(1333, 800)], keep_ratio=True),
                                dict(augmentation="Pad", size_divisor=32),
                            ]),
               input_size=(1024, 1024),
               samples=5000)
    h.test=dict(nms="CombinedNonMaxSuppression",
                pre_nms_size=5000,
                post_nms_size=100, 
                iou_threshold=0.6, 
                score_threshold=0.5)

    return h
