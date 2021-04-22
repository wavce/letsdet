from configs import Config


model = {
    "YOLOv4": [
        #        [from, number, module, args]
        # backbone     
        [-1, 1, "ConvNormActBlock", dict(filters=32, kernel_size=3, strides=1, activation="mish")],  # 0-P1
        [-1, 1, "BottleneckCSP", dict(filters=64, strides=2, dilation_rate=1, expansion=0.5)],       # 1-P1/2
        [-1, 2, "BottleneckCSP", dict(filters=128, strides=2, dilation_rate=1, expansion=0.5)],        # 2-P2/4
        [-1, 8, "BottleneckCSP", dict(filters=256, strides=2, dilation_rate=1, expansion=0.5)],        # 3-P3/8
        [-1, 8, "BottleneckCSP", dict(filters=512, strides=2, dilation_rate=1, expansion=0.5)],        # 4-P4/16
        [-1, 4, "BottleneckCSP", dict(filters=1024, strides=2, dilation_rate=1, expansion=0.5)],       # 5-P5/32

        # spp
        [-1, 1, "Bottleneck", dict(filters=1024, activation=dict(activation="leaky_relu", alpha=0.1), shortcut=False)],      # 6
        [-1, 1, "ConvNormActBlock", dict(filters=512, kernel_size=1, activation=dict(activation="leaky_relu", alpha=0.1))],  # 7
        [-1, 1, "SpatialPyramidPooling", dict(pool_sizes=[5, 9, 13])],                                                       # 8
        [-1, 1, "Bottleneck", dict(filters=1024, activation=dict(activation="leaky_relu", alpha=0.1), shortcut=False)],      # 9
        [-1, 1, "ConvNormActBlock", dict(filters=512, kernel_size=1, activation=dict(activation="leaky_relu", alpha=0.1))],  # 10 (Top-Down: P5)

        # top-down
        [-1, 1, "ConvNormActBlock", dict(filters=256, kernel_size=1, activation=dict(activation="leaky_relu", alpha=0.1))],  # 11
        [-1, 1, "Upsample", dict(size=2, interpolation="nearest")],                                                          # 12
        [4, 1, "ConvNormActBlock", dict(filters=256, kernel_size=1, activation=dict(activation="leaky_relu", alpha=0.1))],   # 13  
        [[-1, -2], 1, "Concat", dict(axis=-1)],                                                    # concat backbone P4      # 14
        [-1, 2, "Bottleneck", dict(filters=512, activation=dict(activation="leaky_relu", alpha=0.1), shortcut=False)],       # 15
        [-1, 1, "ConvNormActBlock", dict(filters=256, kernel_size=1, activation=dict(activation="leaky_relu", alpha=0.1))],  # 16 (Top-Down: P4)
        
        [-1, 1, "ConvNormActBlock", dict(filters=128, kernel_size=1, activation=dict(activation="leaky_relu", alpha=0.1))],  # 17
        [-1, 1, "Upsample", dict(size=2, interpolation="nearest")],                                                          # 18
        [3, 1, "ConvNormActBlock", dict(filters=128, kernel_size=1, activation=dict(activation="leaky_relu", alpha=0.1))],   # 19
        [[-1, -2], 1, "Concat", dict(axis=-1)],                                                    # concat backbone P4      # 20
        # P3
        [-1, 2, "Bottleneck", dict(filters=256, activation=dict(activation="leaky_relu", alpha=0.1), shortcut=False)],       # 21
        [-1, 1, "ConvNormActBlock", dict(filters=128, kernel_size=1, activation=dict(activation="leaky_relu", alpha=0.1))],  # 22
        [-1, 1, "ConvNormActBlock", dict(filters=256, kernel_size=3, activation=dict(activation="leaky_relu", alpha=0.1))],  # 23 (P3/8-small)
        [-1, 1, "Conv", dict(kernel_size=1)],                                                                                # 24

        # bottom-up
        # P4
        [-3, 1, "ConvNormActBlock", dict(filters=256, kernel_size=3, strides=2, activation=dict(activation="leaky_relu", alpha=0.1))],   # 25
        [[-1, -10], 1, "Concat", dict(axis=-1)],  # concat head P4                                                                       # 26      
        [-1, 2, "Bottleneck", dict(filters=512, activation=dict(activation="leaky_relu", alpha=0.1), shortcut=False)],                   # 27
        [-1, 1, "ConvNormActBlock", dict(filters=256, kernel_size=1, activation=dict(activation="leaky_relu", alpha=0.1))],              # 28
        
        [-1, 1, "ConvNormActBlock", dict(filters=512, kernel_size=3, activation=dict(activation="leaky_relu", alpha=0.1))],              # 29 (P4/16-middle)
        [-1, 1, "Conv", dict(kernel_size=1)],                                                                                            # 30

        # P5
        [-3, 1, "ConvNormActBlock", dict(filters=512, kernel_size=3, strides=2, activation=dict(activation="leaky_relu", alpha=0.1))],   # 31
        [[-1, -22], 1, "Concat", dict(axis=-1)],  # concat head P4                                                                       # 32
        [-1, 3, "Bottleneck", dict(filters=1024, activation=dict(activation="leaky_relu", alpha=0.1), shortcut=False)],                  # 33 (P4/16-medium)
        [-1, 1, "Conv", dict(kernel_size=1)],                                                                                            # 34
    ],
    "YOLOv4x-mish":[
        #        [from, number, module, args]
        # backbone     
        [-1, 1, "ConvNormActBlock", dict(filters=32, kernel_size=3, strides=1, activation="mish")],  # 0-P1
        [-1, 1, "ConvNormActBlock", dict(filters=80, kernel_size=3, strides=2, activation="mish")],  # 1-P1/2
        [-1, 1, "Bottleneck", dict(filters=80, expansion=0.5, activation="mish")],                   # 2-P1/2
        [-1, 3, "BottleneckCSP", dict(filters=160, strides=2, dilation_rate=1, expansion=0.5)],      # 3-P2/4
        [-1, 10, "BottleneckCSP", dict(filters=320, strides=2, dilation_rate=1, expansion=0.5)],     # 4-P3/8
        [-1, 10, "BottleneckCSP", dict(filters=640, strides=2, dilation_rate=1, expansion=0.5)],     # 5-P4/16
        [-1, 5, "BottleneckCSP", dict(filters=1280, strides=2, dilation_rate=1, expansion=0.5)],     # 6-P5/32

        # SPP
        [-1, 1, "ConvNormActBlock", dict(filters=640, kernel_size=1, activation="mish")],          # 7
        [-2, 1, "Bottleneck", dict(filters=640, expansion=1, shortcut=False, activation="mish")],  # 8
        [-1, 1, "ConvNormActBlock", dict(filters=640, kernel_size=1, activation="mish")],          # 9
        [-1, 1, "SpatialPyramidPooling", dict(pool_sizes=[5, 9, 13])],                             # 10
        [-1, 2, "Bottleneck", dict(filters=640, expansion=1, shortcut=False, activation="mish")],  # 11
        [[-1, -5], 1, "Concat", dict(axis=-1)],                                                    # 12
        [-1, 1, "ConvNormActBlock", dict(filters=640, kernel_size=1, activation="mish")],          # 13 (Top-Down: P5)

        # top-down
        [-1, 1, "ConvNormActBlock", dict(filters=320, kernel_size=1, activation="mish")],          # 14
        [-1, 1, "Upsample", dict(size=2, interpolation="nearest")],                                # 15
        [-11, 1, "ConvNormActBlock", dict(filters=320, kernel_size=1, activation="mish")],         # 16   
        [[-1, -2], 1, "Concat", dict(axis=-1)],  # concat backbone P4                              # 17
        [-1, 1, "ConvNormActBlock", dict(filters=320, kernel_size=1, activation="mish")],          # 18
        [-1, 1, "ConvNormActBlock", dict(filters=320, kernel_size=1, activation="mish")],          # 19
        [-2, 3, "Bottleneck", dict(filters=320, shortcut=False, expansion=1, activation="mish")],  # 20 
        [[-1, -2], 1, "Concat", dict(axis=-1)],                                                    # 21
        [-1, 1, "ConvNormActBlock", dict(filters=320, kernel_size=1, activation="mish")],          # 22 (Top-Down: P4)
        
        [-1, 1, "ConvNormActBlock", dict(filters=160, kernel_size=1, activation="mish")],          # 23
        [-1, 1, "Upsample", dict(size=2, interpolation="nearest")],                                # 24
        [-21, 1, "ConvNormActBlock", dict(filters=160, kernel_size=1, activation="mish")],         # 25
        [[-1, -2], 1, "Concat", dict(axis=-1)],  # concat backbone P3                              # 26
        [-1, 1, "ConvNormActBlock", dict(filters=160, kernel_size=1, activation="mish")],          # 27 (Top-Down: P3)
        [-1, 1, "ConvNormActBlock", dict(filters=160, kernel_size=1, activation="mish")],          # 28
        [-2, 3, "Bottleneck", dict(filters=160, shortcut=False, expansion=1, activation="mish")],  # 29
        [[-1, -2], 1, "Concat", dict(axis=-1)],                                             # 30
        [-1, 1, "ConvNormActBlock", dict(filters=160, kernel_size=1, activation="mish")],   # 31
        [-1, 1, "ConvNormActBlock", dict(filters=320, kernel_size=3, activation="mish")],   # 32
        [-1, 1, "Conv", dict(kernel_size=1)],                                               # 33

        # bottom-up
        [-3, 1, "ConvNormActBlock", dict(filters=320, kernel_size=3, strides=2, activation="mish")],   # 34
        [[-1, -13], 1, "Concat", dict(axis=-1)],  # concat head P4                                     # 35      
        [-1, 1, "ConvNormActBlock", dict(filters=320, kernel_size=1, activation="mish")],              # 36
        [-1, 1, "ConvNormActBlock", dict(filters=320, kernel_size=1, activation="mish")],              # 37
        [-2, 3, "Bottleneck", dict(filters=320, shortcut=False, expansion=1, activation="mish")],      # 38
        [[-1, -2], 1, "Concat", dict(axis=-1)],                                             # 39
        [-1, 1, "ConvNormActBlock", dict(filters=320, kernel_size=1, activation="mish")],   # 40
        [-1, 1, "ConvNormActBlock", dict(filters=640, kernel_size=3, activation="mish")],   # 41
        [-1, 1, "Conv", dict(kernel_size=1)],                                               # 42

        [-3, 1, "ConvNormActBlock", dict(filters=640, kernel_size=3, strides=2, activation="mish")],   # 43
        [[-1, -31], 1, "Concat", dict(axis=-1)],  # concat head P5                                     # 44
        [-1, 1, "ConvNormActBlock", dict(filters=640, kernel_size=1, activation="mish")],              # 45
        [-1, 1, "ConvNormActBlock", dict(filters=640, kernel_size=1, activation="mish")],              # 45
        [-2, 3, "Bottleneck", dict(filters=640, shortcut=False, expansion=1, activation="mish")],      # 46
        [[-1, -2], 1, "Concat", dict(axis=-1)],                                             # 47
        [-1, 1, "ConvNormActBlock", dict(filters=640, kernel_size=1, activation="mish")],   # 48
        [-1, 1, "ConvNormActBlock", dict(filters=1280, kernel_size=3, activation="mish")],  # 49
        [-1, 1, "Conv", dict(kernel_size=1)],                                               # 50
    ],
     "YOLOv4-tiny":[
        #        [from, number, module, args]
        # backbone     
        [-1, 1, "ConvNormActBlock", dict(filters=32, kernel_size=3, strides=2, activation=dict(activation="leaky_relu", alpha=0.1))],  # 0-P1/2
        [-1, 1, "ConvNormActBlock", dict(filters=64, kernel_size=3, strides=2, activation=dict(activation="leaky_relu", alpha=0.1))],  # 1-P2/4
        [-1, 1, "BottleneckCSPTiny", dict(filters=64, groups=2, group_id=1, activation=dict(activation="leaky_relu", alpha=0.1))],     # 2-P2/4
        [-1, 1, "MaxPool", dict(pool_size=2, strides=2)],                                                                              # 3-P3/8
        [-1, 1, "BottleneckCSPTiny", dict(filters=128, groups=2, group_id=1, activation=dict(activation="leaky_relu", alpha=0.1))],    # 4-P3/8
        [-1, 1, "MaxPool", dict(pool_size=2, strides=2)],                                                                              # 5-P4/16
        [-1, 1, "BottleneckCSPTiny", dict(filters=256, groups=2, group_id=1, activation=dict(activation="leaky_relu", alpha=0.1))],    # 6-P4/16
        [-1, 1, "MaxPool", dict(pool_size=2, strides=2)],                                                                              # 7-P5/32

        # head
        # P5
        [-1, 1, "ConvNormActBlock", dict(filters=512, kernel_size=3, strides=1, activation=dict(activation="leaky_relu", alpha=0.1))],  # 8
        [-1, 1, "ConvNormActBlock", dict(filters=256, kernel_size=1, strides=1, activation=dict(activation="leaky_relu", alpha=0.1))],  # 9
        [-1, 1, "ConvNormActBlock", dict(filters=512, kernel_size=3, strides=1, activation=dict(activation="leaky_relu", alpha=0.1))],  # 10
        [-1, 1, "Conv", dict(kernel_size=1)],                                                                                           # 11

        # P4
        [-3, 1, "ConvNormActBlock", dict(filters=128, kernel_size=1, strides=1, activation=dict(activation="leaky_relu", alpha=0.1))],   # 13
        [-1, 1, "Upsample", dict(size=2, interpolation="nearest")],                                                                       # 14
        [-1, 1, "Concat2", dict(axis=-1)],   # concat to P4                                                                              # 15
        [-1, 1, "ConvNormActBlock", dict(filters=256, kernel_size=3, strides=1, activation=dict(activation="leaky_relu", alpha=0.1))],   # 16
        [-1, 1, "Conv", dict(kernel_size=1)],                                                                                            # 17
    ],
    "YOLOv4-csp": [
        #        [from, number, module, args]
        # backbone     
        [-1, 1, "ConvNormActBlock", dict(filters=32, kernel_size=3, strides=1, activation="mish")],   # 0-P1
        [-1, 1, "ConvNormActBlock", dict(filters=64, kernel_size=3, strides=2, activation="mish")],   # 1-P1/2
        [-1, 1, "Bottleneck", dict(filters=64, dilation_rate=1, expansion=0.5, activation="mish")],   # 2-P1/2
        [-1, 2, "BottleneckCSP", dict(filters=128, strides=2, dilation_rate=1, expansion=0.5)],       # 3-P2/4
        [-1, 8, "BottleneckCSP", dict(filters=256, strides=2, dilation_rate=1, expansion=0.5)],       # 4-P3/8
        [-1, 8, "BottleneckCSP", dict(filters=512, strides=2, dilation_rate=1, expansion=0.5)],       # 5-P4/16
        [-1, 4, "BottleneckCSP", dict(filters=1024, strides=2, dilation_rate=1, expansion=0.5)],      # 6-P5/32

        # SPP
        [-1, 1, "ConvNormActBlock", dict(filters=512, kernel_size=1, activation="mish")],          # 7
        [-2, 1, "Bottleneck", dict(filters=512, expansion=1, shortcut=False, activation="mish")],  # 8
        [-1, 1, "ConvNormActBlock", dict(filters=512, kernel_size=1, activation="mish")],          # 9
        [-1, 1, "SpatialPyramidPooling", dict(pool_sizes=[5, 9, 13])],                             # 10
        [-1, 1, "Bottleneck", dict(filters=512, expansion=1, shortcut=False, activation="mish")],  # 11
        [[-1, -5], 1, "Concat", dict(axis=-1)],                                                    # 12
        [-1, 1, "ConvNormActBlock", dict(filters=512, kernel_size=1, activation="mish")],          # 13 (Top-Down: P5)

        # top-down
        [-1, 1, "ConvNormActBlock", dict(filters=256, kernel_size=1, activation="mish")],          # 14
        [-1, 1, "Upsample", dict(size=2, interpolation="nearest")],                                # 15
        [-11, 1, "ConvNormActBlock", dict(filters=256, kernel_size=1, activation="mish")],         # 16   
        [[-1, -2], 1, "Concat", dict(axis=-1)],  # concat backbone P4                              # 17
        [-1, 1, "ConvNormActBlock", dict(filters=256, kernel_size=1, activation="mish")],          # 18
        [-1, 1, "ConvNormActBlock", dict(filters=256, kernel_size=1, activation="mish")],          # 19
        [-2, 2, "Bottleneck", dict(filters=256, shortcut=False, expansion=1, activation="mish")],  # 20 
        [[-1, -2], 1, "Concat", dict(axis=-1)],                                                    # 21
        [-1, 1, "ConvNormActBlock", dict(filters=256, kernel_size=1, activation="mish")],          # 22 (Top-Down: P4)
        
        [-1, 1, "ConvNormActBlock", dict(filters=128, kernel_size=1, activation="mish")],          # 23
        [-1, 1, "Upsample", dict(size=2, interpolation="nearest")],                                # 24
        [-21, 1, "ConvNormActBlock", dict(filters=128, kernel_size=1, activation="mish")],         # 25
        [[-1, -2], 1, "Concat", dict(axis=-1)],  # concat backbone P3                              # 26
        [-1, 1, "ConvNormActBlock", dict(filters=128, kernel_size=1, activation="mish")],          # 27 (Top-Down: P3)
        [-1, 1, "ConvNormActBlock", dict(filters=128, kernel_size=1, activation="mish")],          # 28
        [-2, 2, "Bottleneck", dict(filters=128, shortcut=False, expansion=1, activation="mish")],  # 29
        [[-1, -2], 1, "Concat", dict(axis=-1)],                                             # 30
        [-1, 1, "ConvNormActBlock", dict(filters=128, kernel_size=1, activation="mish")],   # 31
        [-1, 1, "ConvNormActBlock", dict(filters=256, kernel_size=3, activation="mish")],   # 32
        [-1, 1, "Conv", dict(kernel_size=1)],                                               # 33

        # bottom-up
        [-3, 1, "ConvNormActBlock", dict(filters=256, kernel_size=3, strides=2, activation="mish")],   # 34
        [[-1, -13], 1, "Concat", dict(axis=-1)],  # concat head P4                                     # 35      
        [-1, 1, "ConvNormActBlock", dict(filters=256, kernel_size=1, activation="mish")],              # 36
        [-1, 1, "ConvNormActBlock", dict(filters=256, kernel_size=1, activation="mish")],              # 37
        [-2, 2, "Bottleneck", dict(filters=256, shortcut=False, expansion=1, activation="mish")],      # 38
        [[-1, -2], 1, "Concat", dict(axis=-1)],                                             # 39
        [-1, 1, "ConvNormActBlock", dict(filters=256, kernel_size=1, activation="mish")],   # 40
        [-1, 1, "ConvNormActBlock", dict(filters=512, kernel_size=3, activation="mish")],   # 41
        [-1, 1, "Conv", dict(kernel_size=1)],                                               # 42

        [-3, 1, "ConvNormActBlock", dict(filters=512, kernel_size=3, strides=2, activation="mish")],   # 43
        [[-1, -31], 1, "Concat", dict(axis=-1)],  # concat head P5                                     # 44
        [-1, 1, "ConvNormActBlock", dict(filters=512, kernel_size=1, activation="mish")],              # 45
        [-1, 1, "ConvNormActBlock", dict(filters=512, kernel_size=1, activation="mish")],              # 45
        [-2, 2, "Bottleneck", dict(filters=512, shortcut=False, expansion=1, activation="mish")],      # 46
        [[-1, -2], 1, "Concat", dict(axis=-1)],                                             # 47
        [-1, 1, "ConvNormActBlock", dict(filters=512, kernel_size=1, activation="mish")],   # 48
        [-1, 1, "ConvNormActBlock", dict(filters=1024, kernel_size=3, activation="mish")],  # 49
        [-1, 1, "Conv", dict(kernel_size=1)],                                               # 50
    ],
    "ScaledYOLOv4-p5": [
        # [from, number, module, args]
        # backbone
        [-1, 1, "ConvNormActBlock", dict(filters=32, kernel_size=3, activation="mish")],               # 0
        [-1, 1, "ConvNormActBlock", dict(filters=64, kernel_size=3, strides=2, activation="mish")],    # 1-P1/2
        [-1, 1, "BottleneckCSP2", dict(filters=64)],                                # 2    
        [-1, 1, "ConvNormActBlock", dict(filters=128, kernel_size=3, strides=2, activation="mish")],   # 3-P2/4
        [-1, 3, "BottleneckCSP2", dict(filters=128)],                               # 4
        [-1, 1, "ConvNormActBlock", dict(filters=256, kernel_size=3, strides=2, activation="mish")],   # 5-P3/8
        [-1, 15, "BottleneckCSP2", dict(filters=256)],                              # 6
        [-1, 1, "ConvNormActBlock", dict(filters=512, kernel_size=3, strides=2, activation="mish")],   # 7-P4/16
        [-1, 15, "BottleneckCSP2", dict(filters=512)],                              # 8
        [-1, 1, "ConvNormActBlock", dict(filters=1024, kernel_size=3, strides=2, activation="mish")],  # 9-P5/32
        [-1, 7, "BottleneckCSP2", dict(filters=1024)],                              # 10
        
        # head
        [-1, 1, "SPPCSP", dict(filters=512, pool_sizes=[5, 9, 13])],                # 11
        [-1, 1, "ConvNormActBlock", dict(filters=256, kernel_size=1, strides=1, activation="mish")],   # 12
        [-1, 1, "Upsample", dict(size=2, interpolation="nearest")],                 # 13
        [-6, 1, "ConvNormActBlock", dict(filters=256, kernel_size=1, strides=1, activation="mish")],   # 14 # route backbone P4
        [[-1, -2], 1, "Concat", dict(axis=-1)],                                     # 15
        [-1, 3, "BottleneckCSP2_2", dict(filters=256, shortcut=False)],             # 16
        
        [-1, 1, "ConvNormActBlock", dict(filters=128, kernel_size=1, strides=1, activation="mish")],   # 17
        [-1, 1, "Upsample", dict(size=2, interpolation="nearest")],                 # 18
        [-13, 1, "ConvNormActBlock", dict(filters=128, kernel_size=1, strides=1, activation="mish")],  # 19 # route backbone P3
        [[-1, -2], 1, "Concat", dict(axis=-1)],                                     # 20
        [-1, 3, "BottleneckCSP2_2", dict(filters=128, shortcut=False)],             # 21 (P3/8-small)
        
        [-1, 1, "ConvNormActBlock", dict(filters=256, kernel_size=3, strides=1, activation="mish")],   # 22
        [-2, 1, "ConvNormActBlock", dict(filters=256, kernel_size=3, strides=2, activation="mish")],   # 23
        [[-1, -8], 1, "Concat", dict(axis=-1)],  # concat head P4                   # 24
        [-1, 3, "BottleneckCSP2_2", dict(filters=256, shortcut=False)],             # 25 (P4/16-medium)
        
        [-1, 1, "ConvNormActBlock", dict(filters=512, kernel_size=3, strides=1, activation="mish")],   # 26
        [-2, 1, "ConvNormActBlock", dict(filters=512, kernel_size=3, strides=2, activation="mish")],   # 27
        [[-1, -17], 1, "Concat", dict(axis=-1)],  # concat head P5                  # 28
        [-1, 3, "BottleneckCSP2_2", dict(filters=512, shortcut=False)],             # 29 (P5-large)
        [-1, 1, "ConvNormActBlock", dict(filters=1024, kernel_size=3, strides=1, activation="mish")],  # 30
        
        [[-9, -5, -1], 1, "Detect", dict(anchors="anchors", num_classes="num_classes")]
    ],
    "ScaledYOLOv4-p6": [
        # [from, number, module, args]
        # backbone
        [-1, 1, "ConvNormActBlock", dict(filters=32, kernel_size=3, activation="mish")],               # 0
        [-1, 1, "ConvNormActBlock", dict(filters=64, kernel_size=3, strides=2, activation="mish")],    # 1-P1/2
        [-1, 1, "BottleneckCSP2", dict(filters=64)],                                # 2    
        [-1, 1, "ConvNormActBlock", dict(filters=128, kernel_size=3, strides=2, activation="mish")],   # 3-P2/4
        [-1, 3, "BottleneckCSP2", dict(filters=128)],                               # 4
        [-1, 1, "ConvNormActBlock", dict(filters=256, kernel_size=3, strides=2, activation="mish")],   # 5-P3/8
        [-1, 15, "BottleneckCSP2", dict(filters=256)],                              # 6
        [-1, 1, "ConvNormActBlock", dict(filters=512, kernel_size=3, strides=2, activation="mish")],   # 7-P4/16
        [-1, 15, "BottleneckCSP2", dict(filters=512)],                              # 8
        [-1, 1, "ConvNormActBlock", dict(filters=1024, kernel_size=3, strides=2, activation="mish")],  # 9-P5/32
        [-1, 7, "BottleneckCSP2", dict(filters=1024)],                              # 10
        [-1, 1, "ConvNormActBlock", dict(filters=1024, kernel_size=3, strides=2, activation="mish")],  # 11-P6/64
        [-1, 7, "BottleneckCSP2", dict(filters=1024)],                              # 12
        
        # head
        [-1, 1, "SPPCSP", dict(filters=512, pool_sizes=[5, 9, 13])],                # 13
        [-1, 1, "ConvNormActBlock", dict(filters=512, kernel_size=1, strides=1, activation="mish")],   # 14
        [-1, 1, "Upsample", dict(size=2, interpolation="nearest")],                 # 15
        [-6, 1, "ConvNormActBlock", dict(filters=512, kernel_size=1, strides=1, activation="mish")],   # 16 # route backbone P5
        [[-1, -2], 1, "Concat", dict(axis=-1)],                                     # 17
        [-1, 3, "BottleneckCSP2_2", dict(filters=512, shortcut=False)],             # 18
        
        [-1, 1, "ConvNormActBlock", dict(filters=256, kernel_size=1, strides=1, activation="mish")],   # 19
        [-1, 1, "Upsample", dict(size=2, interpolation="nearest")],                 # 20
        [-13, 1, "ConvNormActBlock", dict(filters=256, kernel_size=1, strides=1, activation="mish")],  # 21 # route backbone P4
        [[-1, -2], 1, "Concat", dict(axis=-1)],                                     # 22
        [-1, 3, "BottleneckCSP2_2", dict(filters=256, shortcut=False)],             # 23 
        
        [-1, 1, "ConvNormActBlock", dict(filters=128, kernel_size=1, strides=1, activation="mish")],   # 24
        [-1, 1, "Upsample", dict(size=2, interpolation="nearest")],                 # 25
        [-20, 1, "ConvNormActBlock", dict(filters=128, kernel_size=1, strides=1, activation="mish")],  # 26 # route backbone P3
        [[-1, -2], 1, "Concat", dict(axis=-1)],                                     # 27
        [-1, 3, "BottleneckCSP2_2", dict(filters=128, shortcut=False)],             # 28
        
        [-1, 1, "ConvNormActBlock", dict(filters=256, kernel_size=3, strides=1, activation="mish")],   # 29  (P3/8)
        [-2, 1, "ConvNormActBlock", dict(filters=256, kernel_size=3, strides=2, activation="mish")],   # 30
        [[-1, -8], 1, "Concat", dict(axis=-1)],                                     # 31
        [-1, 3, "BottleneckCSP2_2", dict(filters=256, shortcut=False)],             # 32 

        [-1, 1, "ConvNormActBlock", dict(filters=512, kernel_size=3, strides=1, activation="mish")],   # 33  (P4/16)
        [-2, 1, "ConvNormActBlock", dict(filters=512, kernel_size=3, strides=2, activation="mish")],   # 34
        [[-1, -17], 1, "Concat", dict(axis=-1)],                                    # 35
        [-1, 3, "BottleneckCSP2_2", dict(filters=512, shortcut=False)],             # 36 

        [-1, 1, "ConvNormActBlock", dict(filters=1024, kernel_size=3, strides=1, activation="mish")],  # 37  (P4/32)
        [-2, 1, "ConvNormActBlock", dict(filters=512, kernel_size=3, strides=2, activation="mish")],   # 38
        [[-1, -26], 1, "Concat", dict(axis=-1)],                                    # 39
        [-1, 3, "BottleneckCSP2_2", dict(filters=512, shortcut=False)],             # 40 
        [-1, 1, "ConvNormActBlock", dict(filters=1024, kernel_size=3, strides=1, activation="mish")],  # 41  (P4/64)
        
        [[-13, -9, -5, -1], 1, "Detect", dict(anchors="anchors", num_classes="num_classes")]
    ],
    "ScaledYOLOv4-p7": [
        # [from, number, module, args]
        # backbone
        [-1, 1, "ConvNormActBlock", dict(filters=32, kernel_size=3, activation="mish")],               # 0
        [-1, 1, "ConvNormActBlock", dict(filters=64, kernel_size=3, strides=2, activation="mish")],    # 1-P1/2
        [-1, 1, "BottleneckCSP2", dict(filters=64)],                                # 2    
        [-1, 1, "ConvNormActBlock", dict(filters=128, kernel_size=3, strides=2, activation="mish")],   # 3-P2/4
        [-1, 3, "BottleneckCSP2", dict(filters=128)],                               # 4
        [-1, 1, "ConvNormActBlock", dict(filters=256, kernel_size=3, strides=2, activation="mish")],   # 5-P3/8
        [-1, 15, "BottleneckCSP2", dict(filters=256)],                              # 6
        [-1, 1, "ConvNormActBlock", dict(filters=512, kernel_size=3, strides=2, activation="mish")],   # 7-P4/16
        [-1, 15, "BottleneckCSP2", dict(filters=512)],                              # 8
        [-1, 1, "ConvNormActBlock", dict(filters=1024, kernel_size=3, strides=2, activation="mish")],  # 9-P5/32
        [-1, 7, "BottleneckCSP2", dict(filters=1024)],                              # 10
        [-1, 1, "ConvNormActBlock", dict(filters=1024, kernel_size=3, strides=2, activation="mish")],  # 11-P6/64
        [-1, 7, "BottleneckCSP2", dict(filters=1024)],                              # 12
        [-1, 1, "ConvNormActBlock", dict(filters=1024, kernel_size=3, strides=2, activation="mish")],  # 13-P6/128
        [-1, 7, "BottleneckCSP2", dict(filters=1024)],                              # 14
        
        # head
        [-1, 1, "SPPCSP", dict(filters=512, pool_sizes=[5, 9, 13])],                # 15
        [-1, 1, "ConvNormActBlock", dict(filters=512, kernel_size=1, strides=1, activation="mish")],   # 16
        [-1, 1, "Upsample", dict(size=2, interpolation="nearest")],                 # 17
        [-6, 1, "ConvNormActBlock", dict(filters=512, kernel_size=1, strides=1, activation="mish")],   # 18 # route backbone P6
        [[-1, -2], 1, "Concat", dict(axis=-1)],                                     # 19
        [-1, 3, "BottleneckCSP2_2", dict(filters=512, shortcut=False)],             # 20
        
        [-1, 1, "ConvNormActBlock", dict(filters=512, kernel_size=1, strides=1, activation="mish")],   # 21
        [-1, 1, "Upsample", dict(size=2, interpolation="nearest")],                 # 22
        [-13, 1, "ConvNormActBlock", dict(filters=512, kernel_size=1, strides=1, activation="mish")],  # 23 # route backbone P5
        [[-1, -2], 1, "Concat", dict(axis=-1)],                                     # 24
        [-1, 3, "BottleneckCSP2_2", dict(filters=512, shortcut=False)],             # 25 
        
        [-1, 1, "ConvNormActBlock", dict(filters=256, kernel_size=1, strides=1, activation="mish")],   # 26
        [-1, 1, "Upsample", dict(size=2, interpolation="nearest")],                 # 27
        [-20, 1, "ConvNormActBlock", dict(filters=256, kernel_size=1, strides=1, activation="mish")],  # 28 # route backbone P4
        [[-1, -2], 1, "Concat", dict(axis=-1)],                                     # 29
        [-1, 3, "BottleneckCSP2_2", dict(filters=256, shortcut=False)],             # 30

        [-1, 1, "ConvNormActBlock", dict(filters=128, kernel_size=1, strides=1, activation="mish")],   # 31
        [-1, 1, "Upsample", dict(size=2, interpolation="nearest")],                 # 32
        [-27, 1, "ConvNormActBlock", dict(filters=128, kernel_size=1, strides=1, activation="mish")],  # 33 # route backbone P3
        [[-1, -2], 1, "Concat", dict(axis=-1)],                                     # 34
        [-1, 3, "BottleneckCSP2_2", dict(filters=128, shortcut=False)],             # 35
        
        [-1, 1, "ConvNormActBlock", dict(filters=256, kernel_size=3, strides=1, activation="mish")],   # 36  (P3/8)
        [-2, 1, "ConvNormActBlock", dict(filters=256, kernel_size=3, strides=2, activation="mish")],   # 37
        [[-1, -8], 1, "Concat", dict(axis=-1)],                                     # 38
        [-1, 3, "BottleneckCSP2_2", dict(filters=256, shortcut=False)],             # 39 

        [-1, 1, "ConvNormActBlock", dict(filters=512, kernel_size=3, strides=1, activation="mish")],   # 40 (P4/16)
        [-2, 1, "ConvNormActBlock", dict(filters=512, kernel_size=3, strides=2, activation="mish")],   # 41
        [[-1, -17], 1, "Concat", dict(axis=-1)],                                    # 42
        [-1, 3, "BottleneckCSP2_2", dict(filters=512, shortcut=False)],             # 43 

        [-1, 1, "ConvNormActBlock", dict(filters=1024, kernel_size=3, strides=1, activation="mish")],  # 44 (P4/32)
        [-2, 1, "ConvNormActBlock", dict(filters=512, kernel_size=3, strides=2, activation="mish")],   # 45
        [[-1, -26], 1, "Concat", dict(axis=-1)],                                    # 46
        [-1, 3, "BottleneckCSP2_2", dict(filters=512, shortcut=False)],             # 47 
        
        [-1, 1, "ConvNormActBlock", dict(filters=1024, kernel_size=3, strides=1, activation="mish")],  # 48 (P4/64)
        [-2, 1, "ConvNormActBlock", dict(filters=512, kernel_size=3, strides=2, activation="mish")],   # 49
        [[-1, -35], 1, "Concat", dict(axis=-1)],                                    # 50
        [-1, 3, "BottleneckCSP2_2", dict(filters=512, shortcut=False)],             # 51 

        [-1, 1, "ConvNormActBlock", dict(filters=1024, kernel_size=3, strides=1, activation="mish")],  # 52  (P4/128)
        
        [[-17, -13, -9, -5, -1], 1, "Detect", dict(anchors="anchors", num_classes="num_classes")]
    ],
}

anchors_dict = {
    "YOLOv4": [
        [12, 16, 19, 36, 40, 28],        
        [36, 75, 76, 55, 72, 146], 
        [142, 110, 192, 243, 459, 401]
    ],
    "YOLOv4x-mish": [
        [12, 16, 19, 36, 40, 28], 
        [36, 75, 76, 55, 72, 146], 
        [142, 110, 192, 243, 459, 401]
    ],
    "YOLOv4-csp": [
        [12, 16, 19, 36, 40, 28], 
        [36, 75, 76, 55, 72, 146], 
        [142, 110, 192, 243, 459, 401]
    ],
    "YOLOv4-tiny": [
        [81, 82, 135, 169, 344, 319], # P5/32
        [10, 14, 23, 27, 37, 58]      # P4/64
    ],
    "ScaledYOLOv4-p5": [
        [13, 17, 31, 25, 24, 51, 61, 45],          # P3/8
        [48, 102, 119, 96, 97, 189, 217, 184],     # P4/16
        [171, 384, 324, 451, 616, 618, 800, 800],  # P5/32
    ],
    "ScaledYOLOv4-p6": [
        [13, 17, 31, 25, 24, 51, 61, 45],            # P3/8
        [61, 45, 48, 102, 119, 96, 97, 189],         # P4/16
        [97, 189, 217, 184, 171, 384, 324, 451],     # P5/32
        [324, 451, 545, 357, 616, 618, 1024, 1024],  # P6/64
    ],
    "ScaledYOLOv4-p7": [
        [13, 17, 22, 25, 27, 66, 55, 41],  # P3/8
        [57, 88, 112, 69, 69, 177, 136, 138],  # P4/16
        [136, 138, 287, 114, 134, 275, 268, 248],  # P5/32
        [268, 248, 232, 504, 445, 416, 640, 640],  # P6/64
        [812, 393, 477, 808, 1070, 908, 1408, 1408],  # P7/128
    ]
}

strides_dict = {
    "YOLOv4": [8, 16, 32],
    "YOLOv4x-mish": [8, 16, 32],
    "YOLOv4-csp": [8, 16, 32],
    "YOLOv4-tiny": [32, 16],
    "ScaledYOLOv4-p5": [8, 16, 32],
    "ScaledYOLOv4-p6": [8, 16, 32, 64],
    "ScaledYOLOv4-p7": [8, 16, 32, 64, 128]
}

default_input_sizes = {
    "YOLOv4": 608,
    "YOLOv4x-mish": 640,
    "YOLOv4-csp": 640,
    "YOLOv4-tiny": 416,
    "ScaledYOLOv4-p5": 896,
    "ScaledYOLOv4-p6": 1280,
    "ScaledYOLOv4-p7": 1536
}

min_level_dict = {
    "YOLOv4": 3,
    "YOLOv4x-mish": 3,
    "YOLOv4-csp": 3,
    "YOLOv4-tiny": 4,
    "ScaledYOLOv4-p5": 3,
    "ScaledYOLOv4-p6": 3,
    "ScaledYOLOv4-p7": 3
}


max_level_dict = {
    "YOLOv4": 5,
    "YOLOv4x-mish": 5,
    "YOLOv4-csp": 5,
    "YOLOv4-tiny": 5,
    "ScaledYOLOv4-p5": 5,
    "ScaledYOLOv4-p6": 6,
    "ScaledYOLOv4-p7": 7
}


depth_multiplier_dict = {
    "YOLOv4": 1.,
    "YOLOv4x-mish": 1.,
    "YOLOv4-csp": 1.,
    "YOLOv4-tiny": 1.,
    "ScaledYOLOv4-p5": 1.,
    "ScaledYOLOv4-p6": 1.,
    "ScaledYOLOv4-p7": 1.
}


width_multiplier_dict = {
    "YOLOv4": 1.,
    "YOLOv4x-mish": 1.,
    "YOLOv4-csp": 1.,
    "YOLOv4-tiny": 1.,
    "ScaledYOLOv4-p5": 1.,
    "ScaledYOLOv4-p6": 1.,
    "ScaledYOLOv4-p7": 1.25
}


def get_yolov4_config(input_size=None,
                      num_classes=80, 
                      num_anchors=3,
                      depth_multiplier=1.,
                      width_multiplier=1.,
                      label_assignment="iou", 
                      name="YOLOv4"):
    h = Config()
    
    if input_size is None:
        input_size = default_input_sizes[name]
    h.detector = name
    h.dtype = "float16"
    h.data_format = "channels_last"
    h.num_classes = num_classes

    h.depth_multiplier = depth_multiplier_dict[name]
    h.width_multiplier = width_multiplier_dict[name]
    
    if name not in model:
        raise ValueError(name + " not in ", list(model.keys()))
    h.model = model[name]
    h.min_level = min_level_dict[name]
    h.max_level = max_level_dict[name]
    h.strides = strides_dict[name]
    h.anchors = anchors_dict[name]
    h.num_anchors = len(anchors_dict[name][0]) // 2
    h.input_size = input_size if isinstance(input_size, (tuple, list)) else (input_size, input_size)

    h.label_assignment = label_assignment
    h.anchor_threshold = 0.2
    h.gr = 1.
        
    h.bbox_loss = dict(loss="CIoULoss", weight=1., reduction="none")  
    h.label_loss = dict(loss="BinaryCrossEntropy", weight=1., from_logits=True, reduction="none")  # .631 if finetuning else weight = 1.0
    h.conf_loss = dict(loss="BinaryCrossEntropy", weight=1., from_logits=True, reduction="none")   # 0.911 if finetuning else weight = 1.
    h.balance = [1., 1., 1.] # [4.0, 1.0, 0.4]   # if num_level == 3 else [4.0, 1.0, 0.4, 0.1]
    h.box_weight = 0.05  # 0.0296 if finetune else 0.05
    h.label_weight = .5  # 0.243 if finetune else 0.5
    h.conf_weight = 1.0   # 0.301 if finetune else 1.0
    
    h.weight_decay = 0.0005
    h.excluding_weight_names = ["predicted"]
    h.train=dict(dataset=dict(dataset="COCODataset",
                              batch_size=8,
                              dataset_dir="/data/bail/COCO",
                              training=True,
                              augmentations=[
                                  dict(augmentation="FlipLeftToRight", probability=0.5),
                                  dict(augmentation="RandomDistortColor"),
                                  dict(augmentation="Resize", img_scale=(0.2, 2), keep_ratio=True),
                                  dict(augmentation="Pad", size_divisor=32)
                              ],
                            #   mixup=dict(alpha=8.0, prob=0.5),
                              mosaic=dict(size=input_size, min_image_scale=0.25, prob=1.),
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
                                dict(Resize=dict(size=(input_size, input_size), strides=32, min_scale=1., max_scale=1.0))
                                # dict(ResizeV2=dict(short_side=800, long_side=1333, strides=64, min_scale=1.0, max_scale=1.0))
                            ]),
               samples=5000)
    # h.test=dict(nms="NonMaxSuppressionWithQuality",
    #             pre_nms_size=5000,
    #             post_nms_size=100, 
    #             iou_threshold=0.6, 
    #             score_threshold=0.5,
    #             sigma=0.5,
    #             nms_type="nms")
    h.test=dict(nms="CombinedNonMaxSuppression",
                pre_nms_size=2000,
                post_nms_size=100, 
                iou_threshold=0.6, 
                score_threshold=0.35)

    return h



