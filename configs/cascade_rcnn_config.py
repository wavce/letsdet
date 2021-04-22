from configs import Config


def get_faster_rcnn_config(num_classes=80):
     h = Config()
    
     h.detector = "FasterRCNN"
     h.dtype = "float16"
     h .num_classes = num_classes
     h.backbone = dict(backbone="ResNet50V1D",
                       convolution="conv2d",
                       dropblock=None, 
                       normalization=dict(normalization="batch_norm", momentum=0.997, epsilon=1e-4, trainable=False),
                       activation=dict(activation="relu"),
                       strides=[2, 2, 2, 2, 2],
                       dilation_rates=[1, 1, 1, 1, 1],
                       output_indices=[2, 3, 4, 5],
                       frozen_stages=[-1, ])
     h.neck=dict(neck="FPN", 
                 convolution="conv2d", 
                 feat_dims=256, 
                 input_dims=[256, 512, 1024, 2048], 
                 min_level=2, 
                 max_level=6, 
                 add_extra_convs=False)
     h.anchors=dict(scales=[[32], [64], [128], [256], [512]], 
                    aspect_ratios=[0.5, 1., 2.0], 
                    strides=[4, 8, 16, 32, 64], 
                    num_anchors=3)
     h.rpn_head=dict(head="RPNHead",
                     convolution="conv2d",
                     normalization=None,
                     activation=dict(activation="relu"),
                     feat_dims=256,
                     dropblock=None,
                     num_classes=2,
                     min_level=2,
                     max_level=6,
                     use_sigmoid=False,
                     bbox_encoder=dict(encoder="Box2Delta", weights=None),
                     bbox_decoder=dict(decoder="Delta2Box", weights=None),
                     assigner=dict(assigner="MaxIoUAssigner", pos_iou_thresh=0.7, neg_iou_thresh=0.3),
                     sampler=dict(sampler="RandomSampler", num_proposals=256, pos_fraction=0.5, add_gt_as_proposals=False),
                     label_loss=dict(loss="CrossEntropy", label_smoothing=0.01, weight=1., from_logits=True, reduction="none"),
                     bbox_loss=dict(loss="SmoothL1Loss", delta=1. / 9., weight=1., reduction="none"))
     h.roi_head=dict(num_stages=3,
                     roi_pooling=dict(roi_pooling="MultiLevelAlignedRoIPooling", cropped_size=7, strides=(4, 8, 16, 32, 64)),
                     bbox_head=[
                         dict(head="Shared2FCRCNNHead",
                              activation=dict(activation="relu"),
                              dropblock=None,
                              fc_dims=1024,
                              num_classes=num_classes,
                              bbox_encoder=dict(encoder="Box2Delta", weights=[10., 10., 5., 5.]),
                              bbox_decoder=dict(decoder="Delta2Box", weights=[10., 10., 5., 5.]),  
                              assigner=dict(assigner="MaxIoUAssigner", pos_iou_thresh=0.5, neg_iou_thresh=0.5,),
                              sampler=dict(sampler="RandomSampler", num_proposals=512, pos_fraction=0.25, add_gt_as_proposals=True),
                              label_loss=dict(loss="CrossEntropy", label_smoothing=0.0, weight=1., from_logits=True, reduction="none"),
                              bbox_loss=dict(loss="SmoothL1Loss", delta=1., weight=1., reduction="none"),
                              use_sigmoid=False,
                              reg_class_agnostic=True),
                         dict(head="Shared2FCRCNNHead",
                              activation=dict(activation="relu"),
                              dropblock=None,
                              fc_dims=1024,
                              num_classes=num_classes,
                              bbox_encoder=dict(encoder="Box2Delta", weights=[20., 20., 10., 10.]),
                              bbox_decoder=dict(decoder="Delta2Box", weights=[20., 20., 10., 10.]),  
                              assigner=dict(assigner="MaxIoUAssigner", pos_iou_thresh=0.6, neg_iou_thresh=0.6,),
                              sampler=dict(sampler="RandomSampler", num_proposals=512, pos_fraction=0.25, add_gt_as_proposals=True),
                              label_loss=dict(loss="CrossEntropy", label_smoothing=0.01, weight=1., from_logits=True, reduction="none"),
                              bbox_loss=dict(loss="SmoothL1Loss", delta=1., weight=1., reduction="none"),
                              use_sigmoid=False,
                              reg_class_agnostic=True),
                         dict(head="Shared2FCRCNNHead",
                              activation=dict(activation="relu"),
                              dropblock=None,
                              fc_dims=1024,
                              num_classes=num_classes,
                              bbox_encoder=dict(encoder="Box2Delta", weights=[1. / 0.033, 1. / 0.022, 1. / 0.067, 1. / 0.067]),
                              bbox_decoder=dict(decoder="Delta2Box", weights=[1. / 0.033, 1. / 0.022, 1. / 0.067, 1. / 0.067]),  
                              assigner=dict(assigner="MaxIoUAssigner", pos_iou_thresh=0.7, neg_iou_thresh=0.7),
                              sampler=dict(sampler="RandomSampler", num_proposals=512, pos_fraction=0.25, add_gt_as_proposals=True),
                              label_loss=dict(loss="CrossEntropy", label_smoothing=0.01, weight=1., from_logits=True, reduction="none"),
                              bbox_loss=dict(loss="SmoothL1Loss", delta=1., weight=1., reduction="none"),
                              use_sigmoid=False,
                              reg_class_agnostic=True)])
     h.weight_decay = 1e-4
     h.train=dict(proposal_layer=dict(pre_nms_size=12000, post_nms_size=2000, max_total_size=2000, iou_threshold=0.7, min_size=0),
                  input_size=(1024, 1024),
                  dataset=dict(dataset="COCODataset",
                               batch_size=2,
                               dataset_dir="/data/bail/COCO",
                               training=True,
                               augmentations=[dict(FlipLeftToRight=dict(probability=0.5)),
                                              dict(RandomDistortColor=dict(probability=1.)),
                                              dict(Resize=dict(size=(1024, 1024), strides=128, min_scale=1.0, max_scale=1.0)),],
                               num_samples=118287,
                               num_classes=num_classes),
                   pretrained_weights_path="/data/bail/pretrained_weights/resnet50_v1d.h5",

                   optimizer=dict(optimizer="SGD", momentum=0.9),
                   mixed_precision=dict(loss_scale=None),  # The loss scale in mixed precision training. If None, use dynamic.
                   gradient_clip_norm=10.0,

                   scheduler=dict(train_epochs=12,
                                  learning_rate_scheduler=dict(scheduler="PiecewiseConstantDecay",
                                                               boundaries=[8, 11],
                                                               values=[0.02, 0.002, 0.0002]),
                                  warmup=dict(warmup_learning_rate=0.0001, steps=8000)),
                   checkpoint_dir="checkpoints/faster_rcnn",
                   summary_dir="logs/faster_rcnn",
                   log_every_n_steps=100,
                   save_ckpt_steps=5000)
     h.val=dict(dataset=dict(dataset="COCODataset", 
                             batch_size=2,  
                             dataset_dir="/data/bail/COCO", 
                             training=False, 
                             augmentations=[dict(Resize=dict(size=(1024, 1024), strides=128, min_scale=1.0, max_scale=1.0))]),
                input_size=(1024, 1024),
                samples=5000, val_every_n_steps=250)
     h.test=dict(proposal_layer=dict(pre_nms_size=6000, post_nms_size=1000, max_total_size=1000, iou_threshold=0.7, min_size=0),
                 pre_nms_size=1000,   # select top_k high confident detections for nms 
                 post_nms_size=100,
                 iou_threshold=0.5,
                 score_threshold=0.05)

     return h
