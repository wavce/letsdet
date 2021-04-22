import itertools
import tensorflow as tf
from ..builder import NECKS
from core.layers import build_activation
from core.layers import build_convolution
from core.layers import build_normalization



class FNode(tf.keras.layers.Layer):
    """A Keras Layer implementing BiFPN Node."""

    def __init__(self,
                 feat_level,
                 inputs_offsets,
                 feat_dims,
                 convolution="conv2d",
                 normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-3, axis=-1, trainable=True), 
                 activation=dict(activation="swish"), 
                 apply_bn_for_resampling=True,
                 conv_after_downsample=False,
                 conv_bn_act_pattern=False,
                 fusion_method="fastattn",
                 data_format="channels_last",
                 name='fnode'):
        super(FNode, self).__init__(name=name)
        self.feat_level = feat_level
        self.inputs_offsets = inputs_offsets
        self.feat_dims = feat_dims
        self.apply_bn_for_resampling = apply_bn_for_resampling
        self.convolution = convolution
        self.normalization = normalization
        self.activation = activation
        self.conv_after_downsample = conv_after_downsample
        self.data_format = data_format
        self.weight_method = fusion_method
        self.conv_bn_act_pattern = conv_bn_act_pattern
        self.resample_layers = []
        self.vars = []

    def fuse_features(self, nodes):
        """Fuse features from different resolutions and return a weighted sum.
        Args:
        nodes: a list of tensorflow features at different levels
        Returns:
        A tensor denoting the fused feature.
        """
        dtype = nodes[0].dtype

        if self.weight_method == 'attn':
            edge_weights = []
            for var in self.vars:
                var = tf.cast(var, dtype=dtype)
                edge_weights.append(var)
            normalized_weights = tf.nn.softmax(tf.stack(edge_weights))
            nodes = tf.stack(nodes, axis=-1)
            new_node = tf.reduce_sum(nodes * normalized_weights, -1)
        elif self.weight_method == 'fastattn':
            edge_weights = []
            for var in self.vars:
                var = tf.cast(var, dtype=dtype)
                edge_weights.append(var)
            weights_sum = tf.add_n(edge_weights)
            nodes = [
                nodes[i] * edge_weights[i] / (weights_sum + 0.0001)
                for i in range(len(nodes))
            ]
            new_node = tf.add_n(nodes)
        elif self.weight_method == 'channel_attn':
            edge_weights = []
            for var in self.vars:
                var = tf.cast(var, dtype=dtype)
                edge_weights.append(var)
            normalized_weights = tf.nn.softmax(tf.stack(edge_weights, -1), axis=-1)
            nodes = tf.stack(nodes, axis=-1)
            new_node = tf.reduce_sum(nodes * normalized_weights, -1)
        elif self.weight_method == 'channel_fastattn':
            edge_weights = []
            for var in self.vars:
                var = tf.cast(var, dtype=dtype)
                edge_weights.append(var)

            weights_sum = tf.add_n(edge_weights)
            nodes = [
                nodes[i] * edge_weights[i] / (weights_sum + 0.0001)
                for i in range(len(nodes))
            ]
            new_node = tf.add_n(nodes)
        elif self.weight_method == 'sum':
            new_node = sum(nodes)  # tf.add_n is not supported by tflite gpu.
        else:
            raise ValueError('unknown weight_method %s' % self.weight_method)

        return new_node

    def _add_wsm(self, initializer):
        for i, _ in enumerate(self.inputs_offsets):
            name = 'WSM' + ('' if i == 0 else '_' + str(i))
            self.vars.append(self.add_weight(initializer=initializer, name=name))

    def build(self, feats_shape):
        for i, input_offset in enumerate(self.inputs_offsets):
            name = 'resample_{}_{}_{}'.format(i, input_offset, len(feats_shape))
            self.resample_layers.append(
                ResampleFeatureMap(
                    feat_level=self.feat_level,
                    target_num_channels=self.feat_dims,
                    normalization=self.normalization,
                    activation=self.activation,
                    apply_bn=self.apply_bn_for_resampling,
                    conv_after_downsample=self.conv_after_downsample,
                    data_format=self.data_format,
                    name=name))
        if self.weight_method == 'attn':
            self._add_wsm('ones')
        elif self.weight_method == 'fastattn':
            self._add_wsm('ones')
        elif self.weight_method == 'channel_attn':
            num_filters = int(self.feat_dims)
            self._add_wsm(lambda: tf.ones([num_filters]))
        elif self.weight_method == 'channel_fastattn':
            num_filters = int(self.feat_dims)
            self._add_wsm(lambda: tf.ones([num_filters]))

        self.op_after_combine = OpAfterCombine(
            convolution=self.convolution,
            normalization=self.normalization,
            activation=self.activation,
            feat_dims=self.feat_dims,
            conv_bn_act_pattern=self.conv_bn_act_pattern,
            data_format=self.data_format,
            name='op_after_combine{}'.format(len(feats_shape)))
        self.built = True
        super(FNode, self).build(feats_shape)

    def call(self, feats, training):
        nodes = []
        for i, input_offset in enumerate(self.inputs_offsets):
            input_node = feats[input_offset]
            input_node = self.resample_layers[i](input_node, training, feats)
            nodes.append(input_node)

        new_node = self.fuse_features(nodes)
        new_node = self.op_after_combine(new_node)

        return feats + [new_node]


class OpAfterCombine(tf.keras.layers.Layer):
    """Operation after combining input features during feature fusiong."""

    def __init__(self,
                 convolution="conv2d",
                 normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-3, axis=-1, trainable=True), 
                 activation=dict(activation="swish"), 
                 feat_dims=64,
                 conv_bn_act_pattern=False,
                 data_format="channels_last",
                 name='op_after_combine'):
        super(OpAfterCombine, self).__init__(name=name)
        self.feat_dims = feat_dims
        self.data_format = data_format
        self.conv_bn_act_pattern = conv_bn_act_pattern

        self.conv_op = build_convolution(
            convolution=convolution,
            filters=feat_dims,
            kernel_size=(3, 3),
            padding='same',
            use_bias=not self.conv_bn_act_pattern,
            data_format=self.data_format,
            name='conv')
        self.bn = build_normalization(**normalization, name='bn')
        self.act = build_activation(**activation)

    def call(self, new_node, training):
        if not self.conv_bn_act_pattern:
            new_node =self.act(new_node)
        new_node = self.conv_op(new_node)
        new_node = self.bn(new_node, training=training)
        if self.conv_bn_act_pattern:
            new_node = self.act(new_node)
        return new_node


class ResampleFeatureMap(tf.keras.layers.Layer):
    """Resample feature map for downsampling or upsampling."""

    def __init__(self,
                 feat_level,
                 target_num_channels,
                 normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-3, axis=-1, trainable=True), 
                 activation=dict(activation="swish"), 
                 apply_bn=False,
                 conv_after_downsample=False,
                 data_format=None,
                 pooling_type=None,
                 upsampling_type=None,
                 name='resample_p0'):
        super(ResampleFeatureMap, self).__init__(name=name)
        self.apply_bn = apply_bn
        self.data_format = data_format
        self.target_num_channels = target_num_channels
        self.feat_level = feat_level
        self.conv_after_downsample = conv_after_downsample
        self.pooling_type = pooling_type or 'max'
        self.upsampling_type = upsampling_type or 'nearest'

        self.conv2d = tf.keras.layers.Conv2D(
            self.target_num_channels, (1, 1),
            padding='same',
            data_format=self.data_format,
            name='conv2d')
        self.bn = build_normalization(**normalization, name='bn')

    def _pool2d(self, inputs, height, width, target_height, target_width):
        """Pool the inputs to target height and width."""
        height_stride_size = int((height - 1) // target_height + 1)
        width_stride_size = int((width - 1) // target_width + 1)
        if self.pooling_type == 'max':
            return tf.keras.layers.MaxPooling2D(
                pool_size=[height_stride_size + 1, width_stride_size + 1],
                strides=[height_stride_size, width_stride_size],
                padding='SAME',
                data_format=self.data_format)(inputs)
        elif self.pooling_type == 'avg':
            return tf.keras.layers.AveragePooling2D(
                pool_size=[height_stride_size + 1, width_stride_size + 1],
                strides=[height_stride_size, width_stride_size],
                padding='SAME',
                data_format=self.data_format)(inputs)
        else:
            raise ValueError('Unsupported pooling type {}.'.format(self.pooling_type))

    def _upsample2d(self, inputs, target_height, target_width):
        return tf.cast(
            tf.image.resize(
                tf.cast(inputs, tf.float32), [target_height, target_width],
                method=self.upsampling_type), inputs.dtype)

    def _maybe_apply_1x1(self, feat, training, num_channels):
        """Apply 1x1 conv to change layer width if necessary."""
        if num_channels != self.target_num_channels:
            feat = self.conv2d(feat)
        if self.apply_bn:
            feat = self.bn(feat, training=training)
        return feat

    def call(self, feat, training, all_feats=None):
        hwc_idx = (2, 3, 1) if self.data_format == 'channels_first' else (1, 2, 3)
        height, width, num_channels = [feat.shape.as_list()[i] for i in hwc_idx]
        if all_feats:
            target_feat_shape = all_feats[self.feat_level].shape.as_list()
            target_height, target_width, _ = [target_feat_shape[i] for i in hwc_idx]
        else:
            # Default to downsampling if all_feats is empty.
            target_height, target_width = (height + 1) // 2, (width + 1) // 2

        # If conv_after_downsample is True, when downsampling, apply 1x1 after
        # downsampling for efficiency.
        if height > target_height and width > target_width:
            if not self.conv_after_downsample:
                feat = self._maybe_apply_1x1(feat, training, num_channels)
            feat = self._pool2d(feat, height, width, target_height, target_width)
            if self.conv_after_downsample:
                feat = self._maybe_apply_1x1(feat, training, num_channels)
        elif height <= target_height and width <= target_width:
            feat = self._maybe_apply_1x1(feat, training, num_channels)
            if height < target_height or width < target_width:
                feat = self._upsample2d(feat, target_height, target_width)
        else:
            raise ValueError(
                'Incompatible Resampling : feat shape {}x{} target_shape: {}x{}'
                .format(height, width, target_height, target_width))

        return feat


@NECKS.register("BiFPN")
class FPNCells(tf.keras.layers.Layer):
    """FPN cells."""

    def __init__(self, 
                 repeats,
                 convolution="separable_conv2d",
                 normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-3, axis=-1, trainable=True), 
                 activation=dict(activation="swish"), 
                 min_level=3,
                 max_level=7,
                 feat_dims=88,
                 fusion_method="fastattn",
                 apply_bn_for_resampling=True,
                 conv_after_downsample=False,
                 conv_bn_act_pattern=False,
                 data_format="channels_last",
                 name='fpn_cells'):
        super(FPNCells, self).__init__(name=name)
        
        self.nodes = get_nodes(min_level, max_level)

        # Feature network.
        self.resample_layers = []  # additional resampling layers.
        for level in range(6, max_level + 1):
            # Adds a coarser level by downsampling the last feature map.
            self.resample_layers.append(
                ResampleFeatureMap(
                    normalization=normalization,
                    activation=activation,
                    feat_level=(level - min_level),
                    target_num_channels=feat_dims,
                    apply_bn=apply_bn_for_resampling,
                    conv_after_downsample=conv_after_downsample,
                    data_format=data_format,
                    name='resample_p%d' % level,
                ))
        
        self.cells = [
            FPNCell(nodes=self.nodes,
                    convolution=convolution, 
                    normalization=normalization,
                    activation=activation,
                    min_level=min_level,
                    max_level=max_level,
                    feat_dims=feat_dims,
                    fusion_method=fusion_method,
                    apply_bn_for_resampling=apply_bn_for_resampling,
                    conv_after_downsample=conv_after_downsample,
                    conv_bn_act_pattern=conv_bn_act_pattern,
                    data_format=data_format,
                    name='cell_%d' % rep)
            for rep in range(repeats)
        ]

        self.min_level = min_level
        self.max_level = max_level

    def call(self, feats, training):
        for resample_layer in self.resample_layers:
            feats.append(resample_layer(feats[-1], training=training))

        for cell in self.cells:
            cell_feats = cell(feats, training)
            min_level = self.min_level
            max_level = self.max_level

        feats = []
        for level in range(min_level, max_level + 1):
            for i, fnode in enumerate(reversed(self.nodes)):
                if fnode['feat_level'] == level:
                    feats.append(cell_feats[-1 - i])
                    break

        return feats


def get_nodes(min_level, max_level):
    # Node id starts from the input features and monotonically increase whenever
    # a new node is added. Here is an example for level P3 - P7:
    #     P7 (4)              P7" (12)
    #     P6 (3)    P6' (5)   P6" (11)
    #     P5 (2)    P5' (6)   P5" (10)
    #     P4 (1)    P4' (7)   P4" (9)
    #     P3 (0)              P3" (8)
    # So output would be like:
    # [
    #   {'feat_level': 6, 'inputs_offsets': [3, 4]},  # for P6'
    #   {'feat_level': 5, 'inputs_offsets': [2, 5]},  # for P5'
    #   {'feat_level': 4, 'inputs_offsets': [1, 6]},  # for P4'
    #   {'feat_level': 3, 'inputs_offsets': [0, 7]},  # for P3"
    #   {'feat_level': 4, 'inputs_offsets': [1, 7, 8]},  # for P4"
    #   {'feat_level': 5, 'inputs_offsets': [2, 6, 9]},  # for P5"
    #   {'feat_level': 6, 'inputs_offsets': [3, 5, 10]},  # for P6"
    #   {'feat_level': 7, 'inputs_offsets': [4, 11]},  # for P7"
    # ]
    num_levels = max_level - min_level + 1
    node_ids = {min_level + i: [i] for i in range(num_levels)}

    level_last_id = lambda level: node_ids[level][-1]
    level_all_ids = lambda level: node_ids[level]
    id_cnt = itertools.count(num_levels)

    nodes = []
    for i in range(max_level - 1, min_level - 1, -1):
        # top-down path.
        nodes.append({
            'feat_level': i,
            'inputs_offsets': [level_last_id(i),
                              level_last_id(i + 1)]
        })
        node_ids[i].append(next(id_cnt))

    for i in range(min_level + 1, max_level + 1):
        # bottom-up path.
        nodes.append({
            'feat_level': i,
            'inputs_offsets': level_all_ids(i) + [level_last_id(i - 1)]
        })
        node_ids[i].append(next(id_cnt))

    return nodes


class FPNCell(tf.keras.layers.Layer):
    """A single FPN cell."""

    def __init__(self, 
                 nodes,
                 convolution="conv2d",
                 normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-3, axis=-1, trainable=True), 
                 activation=dict(activation="swish"), 
                 min_level=3,
                 max_level=7,
                 feat_dims=88,
                 fusion_method="fastattn",
                 apply_bn_for_resampling=True,
                 conv_after_downsample=False,
                 conv_bn_act_pattern=False,
                 data_format="channels_last",
                 name='fpn_cell'):
        super(FPNCell, self).__init__(name=name)

        self.fnodes = []
        for i, fnode_cfg in enumerate(nodes):
            fnode = FNode(feat_level=fnode_cfg['feat_level'] - min_level,
                          inputs_offsets=fnode_cfg['inputs_offsets'],
                          feat_dims=feat_dims,
                          convolution=convolution,
                          normalization=normalization, 
                          activation=activation, 
                          apply_bn_for_resampling=apply_bn_for_resampling,
                          conv_after_downsample=conv_after_downsample,
                          conv_bn_act_pattern=conv_bn_act_pattern,
                          fusion_method=fusion_method,
                          data_format=data_format,
                          name='fnode%d' % i)
            self.fnodes.append(fnode)

    def call(self, feats, training):
        for fnode in self.fnodes:
            feats = fnode(feats, training)
        return feats
