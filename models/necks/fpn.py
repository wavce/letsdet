import tensorflow as tf
from ..common import ConvNormActBlock
from models.builder import NECKS


@NECKS.register("FPN")
class FPN(tf.keras.Model):
    r"""Feature Pyramid Network
    
    Args:
        min_level (int): The min backbone level used to build the feature pyramid. Default: 3 (means C3).
        max_level (int): The max backbone level to build the feature pyramid. Default: 5 (means C5). 
        num_output_levels (int): The number of output levels.
        add_extra_convs (bool): If bool, it decides whether to add conv
            layers on top of the original feature maps. Default to False.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        use_norm_on_lateral (bool): Whether to apply norm in lateral. Default: False.
    """

    def __init__(self,
                 feat_dims,
                 min_level=3,
                 max_level=5,
                 num_output_levels=5,
                 use_norm_on_lateral=False,
                 normalization=None,
                 activation=None,
                 dropblock=None,
                 add_extra_convs=False,
                 add_extra_convs_on_c5=False,
                 relu_before_extra_convs=False,
                 data_format="channels_last",
                 kernel_initializer="he_normal",
                 input_shapes=None,
                 **kwargs):
        super(FPN, self).__init__(**kwargs)

        self.num_backbone_levels = max_level - min_level + 1
        assert self.num_backbone_levels <= num_output_levels
        self.min_level = min_level
        self.max_level = max_level
        self.add_extra_convs_on_c5 = add_extra_convs_on_c5
        self.add_extra_convs = add_extra_convs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.num_outputs_levels = num_output_levels
        self.data_format = data_format

        for l in range(min_level, max_level + 1):
            l_conv = ConvNormActBlock(filters=feat_dims,
                                      kernel_size=1,
                                      normalization=normalization if normalization and use_norm_on_lateral else None,
                                      activation=activation,
                                      data_format=data_format,
                                      dropblock=dropblock,
                                      kernel_initializer=kernel_initializer,
                                      name="lateral_convs/%d" % l)
            fpn_conv = ConvNormActBlock(filters=feat_dims,
                                        kernel_size=3,
                                        normalization=normalization,
                                        kernel_initializer=kernel_initializer,
                                        dropblock=dropblock,
                                        data_format=data_format,
                                        activation=activation,
                                        name="fpn_convs/%d" % l)
            
            setattr(self, "lateral_conv%d" % l, l_conv)
            setattr(self, "fpn_conv%d" % l, fpn_conv)
        
        self.nearest_upsample = tf.keras.layers.UpSampling2D(data_format=data_format)
        ## add extra conv
        self.extra_levels = num_output_levels - self.num_backbone_levels
        if add_extra_convs and self.extra_levels >= 1:
            for l in range(max_level + 1, max_level + self.extra_levels + 1):
                extra_fpn_conv = ConvNormActBlock(filters=feat_dims,
                                                  kernel_size=3,
                                                  strides=2,
                                                  normalization=normalization,
                                                  kernel_initializer=kernel_initializer,
                                                  dropblock=dropblock,
                                                  data_format=data_format,
                                                  activation=activation,
                                                  name="fpn_convs/%d" % l)
                setattr(self, "fpn_conv%d" % l, extra_fpn_conv)

    def call(self, inputs, training=None):
        assert len(inputs) == self.num_backbone_levels, "The length of intputs[%d] not equal the " \
                "number of backbone levels [%d]" % (len(inputs), self.num_backbone_levels)

        # build laterals
        laterals = [
            getattr(self, "lateral_conv%d" % l)(inputs[i], training=training)
            for i, l in enumerate(range(self.min_level, self.max_level + 1))
        ]
        # build top-down path
        for i in range(self.num_backbone_levels-1, 0, -1):
            laterals[i - 1] += self.nearest_upsample(laterals[i])
        
        outputs = [
            getattr(self, "fpn_conv%d" % (i + self.min_level))(laterals[i], training=training) 
            for i in range(self.num_backbone_levels)
        ]

        # add extra levels
        if self.num_outputs_levels > len(outputs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outputs_levels - self.num_backbone_levels):
                    data_format = "NHWC" if self.data_format == "channels_last" else "NCHW"
                    outputs.append(tf.nn.max_pool(outputs[-1], 1, 2, "SAME", data_format=data_format))
            else:
                p5 = outputs[-1]
                if self.add_extra_convs_on_c5:
                    p5 = inputs[-1]
                
                outputs.append(getattr(self, "fpn_conv%d" % (self.max_level + 1))(p5, training=training))

                for l in range(self.max_level + 2, self.num_outputs_levels + self.min_level):
                    fpn_conv = getattr(self, "fpn_conv%d" % l)
                    if self.relu_before_extra_convs:
                        outputs.append(fpn_conv(tf.nn.relu(outputs[-1]), training=training))
                    else:
                        outputs.append(fpn_conv(outputs[-1], training=training))
        
        return outputs

            

        
     

