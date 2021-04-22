from configs.params_dict import ParamsDict


FPN_CONFIG = ParamsDict(default_params={
    "detector": "FPN",
    "dtype": "float32",
    "backbone": {
        "name": "resnet50",
        "convolution": "conv2d",
        "dropblock": None,  # {
            # "dropblock_keep_prob": None,
            # "dropblock_size": None,
        # },
        "normalization": {
            "name": "batch_norm",
            "momentum": 0.997,
            "epsilon": 1e-4,
            "trainable": False,
        },
        "activation": {"activation": "relu"},
        "strides": [2, 2, 2, 2, 2],
        "dilation_rates": [1, 1, 1, 1, 1],
        "output_indices": [2, 3, 4, 5],
        "frozen_stages": [-1, ],
        "frozen_batch_norm": True
    },
    "neck": {
        "neck": "fpn",
        "convolution": "conv2d",
        "feat_dims": 256,
        "normalization":  {
            "normalization": "batch_norm",
            "momentum": 0.9,
            "epsilon": 1e-3,
            "axis": -1,
            "trainable": True
        },
        "activation": {"activation": "relu"},
        "dropblock": None,
        "min_level": 2,
        "max_level": 6,
        "num_repeats": 1,
        "add_extra_conv": False,  # Add extra convolution for neck
        "use_multiplication": False,  # Use multiplication in neck, default False
        "weight_decay": 4e-5,
    },
    "head": {
        "convolution": "conv2d",
        "normalization": {
            "normalization": "batch_norm",
            "momentum": 0.9,
            "epsilon": 1e-3,
            "axis": -1,
            "trainable": True
        },
        "activation": {"activation": "relu"},
        "feat_dims": 256,
        "dropblock": None,
        "min_level": 2,
        "max_level": 6,
        "num_classes": 81,
        "prior": 0.01,
        "weight_decay": 5e-4,
        "num_repeats": 4,
    },
    "anchor": {
        "scales": [[8], [16], [32], [64], [100], [300]],
        "aspect_ratios": [[0.5, 1., 2.0], 
                          [1./3., 0.5, 1., 2.0, 3.0], 
                          [1./3., 0.5, 1., 2.0, 3.0], 
                          [1./3., 0.5, 1., 2.0, 3.0],
                          [0.5, 1., 2.],
                          [0.5, 1., 2.]],
        "stides": [4, 8, 16, 32, 64],
        "min_level": 2,
        "max_level": 6,
    },
    "loss": {
        "label_loss": {
            "loss": "cross_entropy",
            "label_smoothing": 0.0,
            "weight": 1.,
            "from_logits": True,
            "reduction": "none"
        },
        "bbox_loss": {
            "loss": "smooth_l1_loss",
            "delta": 1. / 9.,
            "weight": 1.,
            "reduction": "none"
        },
        "weight_decay": 5e-4,
        "use_sigmoid": True,
    },
    "assigner": {
        "assigner": "max_iou_assigner",
        "pos_iou_thresh": 0.7,
        "neg_iou_thresh": 0.3,
        "min_level": 2,
        "max_level": 6,
    },
    "sampler": {
        "sampler": "random_sampler",
        "num_proposals": 256,
        "pos_fraction": 0.5,
        "add_gt_as_proposals": False
    },
       "train": {
        "dataset": {
            "dataset": "objects365",
            "batch_size": 2,
            "input_size": [300, 300],
            "dataset_dir": "/home/bail/Data/data1/Dataset/Objects365/train",
            "training": True,
            "augmentation": [
                dict(ssd_crop=dict(input_size=[300, 300],
                                   patch_area_range=(0.3, 1.),
                                   aspect_ratio_range=(0.5, 2.0),
                                   min_overlaps=(0.1, 0.3, 0.5, 0.7, 0.9),
                                   max_attempts=100,
                                   probability=.5)),
                # dict(data_anchor_sampling=dict(input_size=[input_size, input_size],
                #                                anchor_scales=(16, 32, 64, 128, 256, 512),
                #                                overlap_threshold=0.7,
                #                                max_attempts=50,
                #                                probability=.5)),
                dict(flip_left_to_right=dict(probability=0.5)),
                dict(random_distort_color=dict(probability=1.))
            ]
        },
        "samples": 12876,
        "num_classes": 366,  # 2 

        "pretrained_weights_path": "/home/bail/Workspace/pretrained_weights/ssd",

        "optimizer": {
            "optimizer": "sgd",
            "momentum": 0.9,
        },
        "lookahead": None,
        "mixed_precision": {
            "loss_scale": None,  # The loss scale in mixed precision training. If None, use dynamic.
        },

        "train_steps": 240000,
        "learning_rate_scheduler": {
            # "learning_rate_scheduler": "piecewise_constant",
            # "initial_learning_rate": initial_learning_rate,
            # "boundaries": boundaries,
            # "values": values
            "learning_rate_scheduler": "cosine",
            "initial_learning_rate": 0.007,
            "steps": 240000 - 24000
        },
        "warmup": {
            "initial_learning_rate": 0.007,
            "warmup_learning_rate": 0.00001,
            "steps": 24000,
        },
        "checkpoint_dir": "checkpoints/ssd",
        "summary_dir": "logs/ssd",

        "gradient_clip_norm": .0,

        "log_every_n_steps": 500,
        "save_ckpt_steps": 10000,
    },
    "val": {
        "dataset": {
            "dataset": "objects365",
            "batch_size": 2,
            "input_size": [300, 300],
            "dataset_dir": "/home/bail/Data/data1/Dataset/Objects365/train",
            "training": False,
            "augmentation": None,
        },
        "samples": 3222,
        "num_classes": 366,
        "val_every_n_steps": 15000,
    }, 
    "postprocess": {
        "max_total_size": 1000,
        "nms_iou_threshold": 0.5,
        "score_threshold": 0.05,
        "use_sigmoid": False,
    }},
    restrictions=[
        "head.num_classes == train.num_classes",
        "head.min_level == neck.min_level",
        "anchor.min_level == head.min_level",
        "assigner.min_lvel == head.min_level",
        "bbox_decoder.bbox_mean == bbox_encoder.bbox_mean",
        "bbox_decoder.bbox_std == bbox_encoder.bbox_std",
        "loss.weight_decay == head.weight_decay",
        "loss.weight_decay == neck.weight_decay",
        "train.dataset.dataset == eval.dataset.dataset",
        "train.warmup.initial_learning_rate == train.learning_rate_scheduler.initial_learning_rate"
])
