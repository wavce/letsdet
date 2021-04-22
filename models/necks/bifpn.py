import tensorflow as tf
from ..builder import NECKS
from core.layers import build_activation
from core.layers import build_convolution
from core.layers import build_normalization
from core.layers import NearestUpsampling2D


class WeightedFusion2(tf.keras.layers.Layer):
    def __init__(self, epsilon=0.0001, **kwargs):
        super(WeightedFusion2, self).__init__(**kwargs)

        self.epsilon = epsilon

    def build(self, input_shape):
        self.w1 = self.add_weight(name="WSM",
                                  shape=[],
                                  initializer=tf.keras.initializers.Ones())
        self.w2 = self.add_weight(name="WSM_1",
                                  shape=[],
                                  initializer=tf.keras.initializers.Ones())
        
        super(WeightedFusion2, self).build(input_shape)

    def call(self, inputs, training=None):
        w1 = tf.nn.relu(self.w1)
        w2 = tf.nn.relu(self.w2)
        divisor = tf.add_n([w1, w2]) + self.epsilon
        outputs = tf.add_n([w1 * inputs[0], w2 * inputs[1]]) * (1. / divisor)
        
        return outputs


class WeightedFusion3(tf.keras.layers.Layer):
    def __init__(self, epsilon=0.0001, **kwargs):
        super(WeightedFusion3, self).__init__(**kwargs)

        self.epsilon = epsilon

    def build(self, input_shape):
        self.w1 = self.add_weight(name="WSM",
                                  shape=[],
                                  initializer=tf.keras.initializers.Ones())
        self.w2 = self.add_weight(name="WSM_1",
                                  shape=[],
                                  initializer=tf.keras.initializers.Ones())
        self.w3 = self.add_weight(name="WSM_2",
                                  shape=[],
                                  initializer=tf.keras.initializers.Ones())
        super(WeightedFusion3, self).build(input_shape)

    def call(self, inputs, training=None):
        w1 = tf.nn.relu(self.w1)
        w2 = tf.nn.relu(self.w2)
        w3 = tf.nn.relu(self.w3)
        divisor = tf.add_n([w1, w2, w3]) + self.epsilon
        outputs = tf.add_n([w1 * inputs[0], w2 * inputs[1], w3 * inputs[2]]) * (1. / divisor)
        
        return outputs


def resample_feature_map(feat, 
                         target_width, 
                         target_num_channels,
                         convolution="separable_conv2d",
                         normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-3, axis=-1, trainable=True),
                         activation=dict(activation="swish"),
                         pool_type=None, 
                         apply_bn=True, 
                         name="resample"):
    """Resample input feature map to have target number of channels and width.""" 
    _, _, width, num_channels = tf.keras.backend.int_shape(feat)
    if width > target_width:
        if num_channels != target_num_channels:
            feat = build_convolution("conv2d",
                                     filters=target_num_channels,
                                     kernel_size=(1, 1),
                                     padding="same",
                                     name=name + "/conv2d")(feat)
            if apply_bn:
                feat = build_normalization(**normalization, name=name+"/bn")(feat)
        strides = int(width // target_width)

        # if strides >= 2:
        #     feat = tf.keras.layers.ZeroPadding2D(
        #         padding=imagenet_utils.correct_pad(feat, strides + 1),
        #         name=name + '/stem/conv_pad')(feat)
        #     pad = "valid"
        # else:
        #     pad = "same"
        if pool_type == "max" or pool_type is None:
            feat = tf.keras.layers.MaxPool2D(pool_size=strides+1,
                                             strides=[strides, strides],
                                             padding="same",
                                             name=name + "/max_pool")(feat)
        elif pool_type == "avg":
            feat = tf.keras.layers.AvgPool2D(pool_size=strides+1,
                                             strides=[strides, strides],
                                             padding="same",
                                             name=name + "/avg_pool")(feat)
        else:
            raise ValueError("Unknown pooling type: {}".format(pool_type))
    else:
        if num_channels != target_num_channels:
            feat = build_convolution("conv2d",
                                     filters=target_num_channels,
                                     kernel_size=(1, 1),
                                     padding="same",
                                     name=name + "/conv2d")(feat)
            if apply_bn:
                feat = build_normalization(**normalization, name=name+"/bn")(feat)
        if width < target_width:
            scale = target_width // width
            feat = NearestUpsampling2D(scale=scale, name=name + "/nearest_upsampling")(feat)
    
    return feat


def _verify_feats_size(feats, input_size, min_level, max_level):
    expected_output_width = [
        int(input_size[1] / 2**l) for l in range(min_level, max_level + 1)
    ]
    expected_output_height = [
        int(input_size[0] / 2**l) for l in range(min_level, max_level + 1)
    ]
    for cnt, (height, width) in enumerate(zip(expected_output_height, expected_output_width)):
        if feats[cnt].shape[2] != width or feats[cnt].shape[1] != height:
            raise ValueError('feats[{}] has shape {} but its width should be {}.'
                            '(input_size: {}, min_level: {}, max_level: {}.)'.format(
                                cnt, feats[cnt].shape, (height, width), input_size, min_level,
                                max_level))


def build_bifpn_layer(feats, 
                      input_size, 
                      feat_dims,
                      convolution="separable_conv2d",
                      normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-3, axis=-1, trainable=True),
                      activation=dict(activation="swish"),
                      min_level=3,
                      max_level=7,
                      pool_type=None, 
                      apply_bn=True,
                      fusion_type="weighted_sum", 
                      name="fpn_cells",
                      **kwargs):
    F = lambda x: 1.0 / (2 ** x)  # Resolution size for a given feature level.
    node_configs = [
        {"width_ratio": F(6), "inputs_offsets": [3, 4]},
        {"width_ratio": F(5), "inputs_offsets": [2, 5]},
        {"width_ratio": F(4), "inputs_offsets": [1, 6]},
        {"width_ratio": F(3), "inputs_offsets": [0, 7]},
        {"width_ratio": F(4), "inputs_offsets": [1, 7, 8]},
        {"width_ratio": F(5), "inputs_offsets": [2, 6, 9]},
        {"width_ratio": F(6), "inputs_offsets": [3, 5, 10]},
        {"width_ratio": F(7), "inputs_offsets": [4, 11]},
    ]

    num_output_connections = [0 for _ in feats]
    for i, fnode in enumerate(node_configs):
        nodes = []
        node_name = name + "/fnode{}".format(i)
        new_node_width = int(fnode["width_ratio"] * input_size[1])
        for idx, input_offset in enumerate(fnode["inputs_offsets"]):
            input_node = feats[input_offset]
            num_output_connections[input_offset] += 1
            input_node = resample_feature_map(
                input_node,
                new_node_width,
                feat_dims,
                convolution=convolution,
                normalization=normalization,
                activation=activation,
                pool_type=pool_type,
                apply_bn=apply_bn,
                name=node_name + "/resample_{}_{}_{}".format(idx, input_offset, len(feats)))
            nodes.append(input_node)
        if fusion_type == "weighted_sum":
            if len(fnode["inputs_offsets"]) == 2:
                new_node = WeightedFusion2(name=node_name)(nodes)
            if len(fnode["inputs_offsets"]) == 3:
                new_node = WeightedFusion3(name=node_name)(nodes)
        elif fusion_type == "sum":
            new_node = tf.keras.layers.Add(name=node_name)(nodes)
        else:
            raise  ValueError("Unknown fusion type: {}".format(fusion_type))
        
        new_node_name = node_name + "/op_after_combine{}".format(len(feats))
        new_node = build_activation(
            **activation, name=new_node_name + "/" + activation["activation"])(new_node)

        new_node = build_convolution(convolution,
                                     filters=feat_dims,
                                     kernel_size=(3, 3),
                                     padding="same",
                                     name=new_node_name + "/conv")(new_node)
        new_node = build_normalization(
            **normalization, name=new_node_name + "/bn" )(new_node)
        # new_node = build_activation(
        #     **activation, name=new_node_name + "/" + activation["activation"] + "_1")(new_node)
        
        feats.append(new_node)
        num_output_connections.append(0)
    
    outputs = []
    for level in range(min_level, max_level + 1):
        for i, fnode in enumerate(reversed(node_configs)):
            if fnode["width_ratio"] == F(level):
                outputs.append(feats[-1 - i])
                break

    return outputs


@NECKS.register("BiFPN")
def BiFPN(input_size, 
          input_dims,
          feat_dims,
          repeats,
          normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-3, axis=-1, trainable=True),
          activation=dict(activation="swish"),
          min_level=3,
          max_level=7,
          pool_type=None, 
          apply_bn=True, 
          fusion_type="weighted_sum",
          name="fpn_cells",
          **kwargs):
    inputs = [
        tf.keras.Input(shape=(input_size[0] // (2 ** l), input_size[1] // (2 ** l), dim), 
                       name="neck_input_%d" % l) 
        for dim, l in zip(input_dims, list(range(min_level, max_level + 1)))
        ] 
    num_inputs = len(inputs)
    feats = [inp for inp in inputs]
    for i in range(min_level + num_inputs, max_level + 1):
        _, _, w, _ = tf.keras.backend.int_shape(feats[-1])
        feats.append(resample_feature_map(feats[-1],
                                          target_width=w // 2,
                                          target_num_channels=feat_dims,
                                          convolution=convolution,
                                          normalization=normalization,
                                          activation=activation,
                                          pool_type=pool_type,
                                          apply_bn=apply_bn,
                                          name="resample_p%d" % i))
    _verify_feats_size(feats, input_size, min_level, max_level)

    for rep in range(repeats):
        feats = build_bifpn_layer(feats,
                                  input_size=input_size,
                                  feat_dims=feat_dims,
                                  convolution=convolution,
                                  normalization=normalization,
                                  activation=activation,
                                  min_level=min_level,
                                  max_level=max_level,
                                  pool_type=pool_type,
                                  apply_bn=apply_bn,
                                  fusion_type=fusion_type,
                                  name=name + "/cell_{}".format(rep))
        _verify_feats_size(feats, input_size, min_level, max_level)
    
    return tf.keras.Model(inputs=inputs, outputs=feats)


class Print(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Print, self).__init__(**kwargs)
    
    def call(self, inputs):
        tf.print(inputs)
        return inputs
