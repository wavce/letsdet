from .base_config import Config


def get_centernet_config(num_classes=80):
    h = Config()

    input_size = 512
    downsample_ratio = 4
    batch_size = 16
    h.detector = "CenterNet"
    h.dtype = "float16"
    h.data_format = "channels_last"
    h.input_shape = (input_size, input_size, 3)
    h.num_classes = num_classes
    # h.backbone = dict(backbone="HourglassNet",
    #                   normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
    #                   activation=dict(activation="relu"),
    #                   frozen_stages=(-1, ),
    #                   output_indices=(1, 2),
    #                   dropblock=None)
    h.backbone = dict(backbone="DLA34",
                      normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                      activation=dict(activation="relu"),
                      return_levels=False,
                      strides=(2, 2, 2, 2, 2), 
                      dilation_rates=(1, 1, 1, 1, 1), 
                      frozen_stages=(-1, ), 
                      dropblock=None)
    h.neck = dict(neck="DLAUp",
                  filters=[16, 32, 64, 128, 256, 512],
                  downsample_ratio=downsample_ratio,
                  data_format="channels_last", 
                  normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                  activation=dict(activation="relu"),
                  kernel_initializer="he_normal")
    h.head=dict(head="CenterHeatmapHead",
                downsample_ratio=downsample_ratio,
                normalization=None,
                activation=dict(activation="relu"),
                num_stacks=1,  ## if backbone is houglass, num_stacks = 2
                feat_dims=256,
                dropblock=None,
                repeats=1,
                assigner=dict(assigner="CenterHeatmapAssigner", strides=downsample_ratio),
                hm_loss=dict(loss="ModifiedFocalLoss", alpha=2.0, beta=4., weight=1.),
                wh_loss=dict(loss="RegL1Loss", weight=0.1),
                reg_loss=dict(loss="RegL1Loss", weight=1.0))
    
    h.weight_decay = 0.0001
    h.excluding_weight_names = []
    h.train=dict(dataset=dict(dataset="COCODataset",
                              batch_size=batch_size,
                              dataset_dir="/data/bail/COCO",
                              training=True,
                              augmentations=[
                                    dict(augmentation="FlipLeftToRight", probability=0.5),
                                    dict(augmentation="RandomDistortColor"),
                                    dict(augmentation="Resize", img_scale=(0.2, 2), multiscale_mode="range", keep_ratio=True),
                                    dict(augmentation="RandCropOrPad", size=(input_size, input_size), clip_box_base_center=False),
                                ],
                              num_samples=118287,
                              num_classes=num_classes),
                  pretrained_weights_path="/data/bail/pretrained_weights/dla34.h5",
                  optimizer=dict(optimizer="Adam"),
                  gradient_clip_norm=.0,
                  scheduler=dict(train_epochs=120,
                                 learning_rate_scheduler=dict(scheduler="PiecewiseConstantDecay",
                                                              boundaries=[80, 100],
                                                              values=[0.001, 0.0001, 0.000012]),
                                #  learning_rate_scheduler=dict(scheduler="CosineDecay", initial_learning_rate=0.02),
                                 warmup=dict(warmup_learning_rate=0.0, steps=800)),
                  checkpoint_dir="checkpoints/centernet_dla34",
                  summary_dir="logs/centernet_dla34",
                  log_every_n_steps=100)
    h.val=dict(dataset=dict(dataset="COCODataset", 
                            batch_size=1,  
                            dataset_dir="/data/bail/COCO", 
                            training=False, 
                            augmentations=[
                                dict(augmentation="Resize", img_scale=[(1333, input_size)], keep_ratio=True),
                                dict(augmentation="Pad", size_divisor=32)
                            ]),
               samples=5000)
    h.test=dict(topk=100, score_threshold=0.1)

    return h