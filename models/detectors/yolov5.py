import math
import numpy as np
import tensorflow as tf
from core import build_nms
from core import build_loss
from utils import box_utils
from models.builder import DETECTORS
from core.layers import build_activation
from core.losses.iou_loss import compute_iou


def hard_swish(inputs):
    with tf.name_scope("hard_swish"):
        return inputs * tf.nn.relu6(inputs + np.float32(3)) * (1. / 6.)


def fuse_conv_and_bn(conv, bn, input_shape):
    fusedconv = tf.keras.layers.Conv2D(
        filters=conv.filters, 
        kernel_size=conv.strides,
        strides=conv.strides,
        padding=conv.padding,
        use_bias=conv.use_bias)
    fusedconv.build(input_shape)
    
    # prepare filters
    

class DWConv(tf.keras.Model):
    def __init__(self, filters, kernel_size=1, strides=1, groups=1, activation=True, name="conv"):
        super(DWConv, self).__init__(name=name)

        padding = "same"
        
        if strides != 1:
            p = (kernel_size - 1) // 2
            self.pad = tf.keras.layers.ZeroPadding2D((p, p))
            padding = "valid"
        
        self.conv = tf.keras.layers.DepthwiseConv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides, 
            padding=padding,
            use_bias=False,
            groups=groups,
            name="conv")
        self.bn = tf.keras.layers.BatchNormalization(axis=-1, name="bn")
        self.activation = activation
        
    def call(self, inputs, training=None):
        if hasattr(self, "pad"):
            x = self.pad(inputs)
        else:
            x = inputs
        x = self.conv(x)
        x = self.bn(x, training)
        
        if self.activation:
            x = hard_swish(x)

        return x
    
    def fuse_call(self, inputs):
        x = self.conv(inputs)
        if self.activation:
            x = hard_swish(x)
        
        return x


class Conv(tf.keras.Model):
    def __init__(self, filters, kernel_size=1, strides=1, groups=1, activation=True, name="conv"):
        super(Conv, self).__init__(name=name)
        
        p = (kernel_size - 1) // 2
        if p >= 1:
            self.pad = tf.keras.layers.ZeroPadding2D((p, p))

        self.conv = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides, 
            padding="valid",
            use_bias=False,
            groups=groups,
            name="conv")
        self.bn = tf.keras.layers.BatchNormalization(axis=-1, name="bn")
        # self.act_fn = tf.keras.layers.LeakyReLU(alpha=0.1) if activation else tf.identity
        self.act_fn = hard_swish if activation else tf.identity
        
    def call(self, inputs, training=None):
        if hasattr(self, "pad"):
            x = self.pad(inputs)
        else:
            x = inputs
        x = self.conv(x)
        x = self.bn(x, training)
        
        x = self.act_fn(x)
        
        return x
    
    def fuse_call(self, inputs):
        x = self.conv(inputs)
        x = self.act_fn(x)
        
        return x


class Bottleneck(tf.keras.Model):
    def __init__(self, in_filters, out_filters, shortcut=True, groups=1, expansion=0.5, name="bottleneck"):
        super(Bottleneck, self).__init__(name=name)
        
        hidden_filters = int(out_filters * expansion)
        self.cv1 = Conv(hidden_filters, 1, 1, name="cv1")
        self.cv2 = Conv(out_filters, 3, 1, groups=groups, name="cv2")
        
        self.add = shortcut and in_filters == out_filters
        
    def call(self, inputs, training=None):
        x = self.cv1(inputs, training=training)
        x = self.cv2(x, training=training)
        
        if self.add:
            x += inputs
        
        return x


class BottleneckCSP(tf.keras.Model):
    """
    CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    
    Args:
        filters(int): input filters.
        n(int): the number of standard bottleneck.
        groups(int): the group in grou conv.
        expansion(int): the expansion rate in bottleneck.
        name(str): the module name.
    """
    def __init__(self, filters, n=1, shortcut=True, groups=1, expansion=0.5, name="bottleneck_csp"):
        super(BottleneckCSP, self).__init__(name=name)
        
        mid_filters = int(filters * expansion)
        self.cv1 = Conv(mid_filters, 1, 1, name="cv1")
        self.cv2 = tf.keras.layers.Conv2D(
            filters=mid_filters, kernel_size=1, strides=1, use_bias=False, name="cv2")
        self.cv3 = tf.keras.layers.Conv2D(
            filters=mid_filters, kernel_size=1, strides=1, use_bias=False, name="cv3")
        self.cv4 = Conv(filters, 1, 1, name="cv4")
        self.bn = tf.keras.layers.BatchNormalization(axis=-1, name="bn")
        self.act = tf.keras.layers.LeakyReLU(alpha=0.1, name="act")
        self.m = tf.keras.Sequential([
            Bottleneck(mid_filters, mid_filters, shortcut, groups, 1., name=self.name + "/m/%d" % i) 
            for i in range(n)], name="m")
        
    def call(self, inputs, training=None):
        x1 = self.cv3(self.m(self.cv1(inputs, training=training), training=training))
        x2 = self.cv2(inputs)
        
        x = tf.concat([x1, x2], axis=-1)
        x = self.bn(x, training=training)
        x = self.act(x)
        x = self.cv4(x, training=training)
        
        return x
    

class SpatialPyramidPooling(tf.keras.Model):
    def __init__(self, filters, pool_sizes=(5, 9, 13), name="spp"):
        super(SpatialPyramidPooling, self).__init__(name=name)
        
        mid_filters = filters // 2
        self.cv1 = Conv(mid_filters, 1, 1, name="cv1")
        self.cv2 = Conv(filters, 1, 1, name="cv2")
        self.m = [
            tf.keras.layers.MaxPool2D(pool_size=ps, strides=1, padding="same") 
            for ps in pool_sizes
            ]
    
    def call(self, inputs, training=None):
        x = self.cv1(inputs, training=training)
        x = tf.concat([x] + [m(x) for m in self.m], axis=-1)
        x = self.cv2(x, training=training)
        
        return x
    

class Focus(tf.keras.Model):
    def __init__(self, filters, kernel_size=1, strides=1, groups=1, activation=True, name="focus"):
        super(Focus, self).__init__(name=name)
        
        self.conv = Conv(filters, kernel_size, strides, groups, activation, "conv")
        
    def call(self, inputs, training=None):
        x = tf.concat([inputs[:, ::2, ::2, :], 
                       inputs[:, 1::2, ::2, :], 
                       inputs[:, ::2, 1::2, :], 
                       inputs[:, 1::2, 1::2, :]], axis=-1)
        x = self.conv(x, training=training)
       
        return x


class CrossConv(tf.keras.Model):
    """
    Cross Convolution Downsample
    """
    def __init__(self, 
                 in_filters, 
                 out_filters, 
                 kernel_size=3, 
                 strides=1, 
                 groups=1, 
                 expansion=1.0, 
                 shortcut=False, 
                 name="cross_conv"):
        super(CrossConv, self).__init__(name=name)
        
        mid_filters = int(out_filters * expansion)
        
        self.cv1 = Conv(mid_filters, (1, kernel_size), (1, strides), name="cv1")
        self.cv2 = Conv(out_filters, (kernel_size, 1), (strides, 1), groups=groups, name="cv2")
        self.add = shortcut and in_filters == out_filters
    
    def call(self, inputs, training=None):
        x = self.cv1(inputs, training=training)
        x = self.cv2(x, training=training)
        
        if self.add:
            x += inputs
        
        return x
    

class C3(tf.keras.Model):
    def __init__(self, filters, n=1, shortcut=True, groups=1, expansion=0.5, name="c3"):
        super(C3, self).__init__(name=name)

        mid_filters = int(filters * expansion)
        
        self.cv1 = Conv(mid_filters, 1, 1, name="cv1")
        self.cv2 = tf.keras.layers.Conv2D(
            filters=mid_filters, kernel_size=1, strides=1, use_bias=False, name="cv2")
        self.cv3 = tf.keras.layers.Conv2D(
            filters=mid_filters, kernel_size=1, strides=1, use_bias=False, name="cv3")
        self.cv4 = Conv(filters, 1, 1, name="cv4")
        self.bn = tf.keras.layers.BatchNormalization(axis=-1, name="bn")
        self.act = tf.keras.layers.LeakyReLU(alpha=0.1, name="act")
        self.m = tf.keras.Sequential([
            CrossConv(mid_filters, mid_filters, 3, 1, groups, 1., shortcut, name=str(i)) 
            for i in range(n)], name="m")
        
    def call(self, inputs, training=None):
        x1 = self.cv1(inputs, training=training)
        x1 = self.m(x1, training=training)
        x1 = self.cv3(x1)
        
        x2 = self.cv2(inputs)
        
        x = tf.concat([x1, x2], axis=1)
        x = self.bn(x, training=training)
        x = self.act(x)
        x = self.cv4(x, training=training)
        
        return x
    

class WeightedSum(tf.keras.Model):
    def __init__(self, num_layers, weight=False, name="sum"):
        super(WeightedSum, self).__init__(name=name)
        
        self.weight = weight
        self.iter = range(num_layers - 1)
        self.num_layers = num_layers
        
    def build(self, input_shape):
        if self.weight:
            self.w = self.add_weight(
                name="WSM",
                shape=[self.num_layers - 1],
                initializer=tf.keras.initializers.Constant(tf.range(1., self.num_layers)))
    
    def call(self, inputs, training=None):
        x = inputs[0]
        
        if self.weight:
            w = tf.nn.sigmoid(self.w) * 2
            for i in self.iter:
                x = x + inputs[i + 1] * w[i]
        else:
            for i in self.iter:
                x = x + inputs[i + 1]
        
        return x
    

class GhostConv(tf.keras.Model):
    """
    Ghost Convolution https://github.com/huawei-noah/ghostnet
    """
    def __init__(self, filters, kernel_size=1, strides=1, groups=1, activation=True, name="ghost_conv"):
        super(GhostConv, self).__init__(name=name)
        
        mid_filters = filters // 2
        
        self.cv1 = Conv(mid_filters, kernel_size, strides, groups, activation, "cv1")
        self.cv2 = Conv(mid_filters, 5, 1, groups, activation, "cv2")
        
    def call(self, inputs, training=None):
        x1 = self.cv1(inputs, traning=training)
        x = self.cv2(x1, training=training)
        x = tf.concat([x1, x], -1)
        
        return x
    
    
class GhostBottleneck(tf.keras.Model):
    """
    Ghost Bottleneck https://github.com/huawei-noah/ghostnet
    """
    def __init__(self, filters, kernel_size, strides, name="ghost"):
        super(GhostBottleneck, self).__init__(name=name)
        
        mid_filters = filters // 2
        
        self.conv = tf.keras.Sequential(name="conv")
        self.conv.add(GhostConv(mid_filters, 1, 1, name="0"))
        if strides == 2:
            self.conv.add(DWConv(mid_filters, kernel_size, strides, activation=False, name="1"))
        self.conv.add(GhostConv(filters, 1, 1, activation=False, name="2"))
        
        self.shortcut = tf.keras.Sequential(name="shortcut")
        self.shortcut.add(DWConv(filters, kernel_size, strides, activation=False, name="0"))
        if strides == 2:
            self.shortcut.add(Conv(filters, 1, 1, activation=False, name="1"))
    
    def call(self, inputs, traning=None):
        return self.conv(inputs, traning=traning) + self.shortcut(inputs, traning=traning)


class MixConv2d(tf.keras.Model):
    """Mixed Depthwise Conv https://arxiv.org/abs/1907.0959"""
    def __init__(self, filters, kernel_size=(1, 3), strides=1, equal_filters=True, name="mix_conv2d"):
        super(MixConv2d, self).__init__(name=name)
        
        groups = len(kernel_size)
        
        if equal_filters: # equal filters per group
            i = np.floor(np.linspace(0, groups - 1E-6, filters))
            mid_filters = [(i == g).sum() for g in range(groups)]
        else:  # equal weight.size() per group
            b = [filters] + [0] * groups
            a = np.eye(groups + 1, groups, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(kernel_size) ** 2
            a[0] = 1
            mid_filters = np.linalg.lstsq(a, b, rcond=None)[0].round()
        
        self.m = [
            tf.keras.layers.Conv2D(
                filters=int(mid_filters[g]),
                kernel_size=int(kernel_size[g]),
                strides=strides,
                padding="same",
                use_bias=False) for g in range(groups)
        ]
        self.bn = tf.keras.layers.BatchNormalization(axis=-1, name="bn")
        self.act = tf.keras.layers.LeakyReLU(alpha=0.1, name="act")
    
    def call(self, inputs, training=None):
        x = tf.concat([m(inputs) for m in self.m], -1)
        x = self.bn(x, training=training)
        x = self.act(x)
        x += inputs
        
        return x
            
            
class Detect(tf.keras.Model):
    def __init__(self, num_classes=80, anchors=(), strides=(8, 16, 32), name="detect"):
        super(Detect, self).__init__(name=name)
        
        self.num_classes = num_classes
        self.num_outputs = num_classes + 5
        
        self.num_layers = len(anchors)  # number of detection layers
        self.num_anchors = len(anchors[0]) // 2 # number of anchors per layer
        self.anchors = anchors
        self.strides = strides
        
        biases = []
        for i in range(self.num_layers):
            s = strides[i]
            bias = np.zeros([self.num_anchors, self.num_outputs])
            bias[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            bias[:, 5:] += math.log(0.6 / (num_classes - 0.99)) # cls
            bias = np.reshape(bias, [-1])
            biases.append(bias)
        self.m = [
            tf.keras.layers.Conv2D(
                filters=self.num_anchors * self.num_outputs, 
                kernel_size=1, 
                bias_initializer=tf.keras.initializers.Constant(biases[i]),
                name="m/%d" % i) 
            for i in range(self.num_layers)
        ]
    
    def call(self, inputs):
        outputs = []
        for i in range(self.num_layers):
            x = self.m[i](inputs[i])
            shape = tf.shape(x)
            bs, ny, nx = shape[0], shape[1], shape[2]
            x = tf.reshape(x, [bs, ny, nx, self.num_anchors, self.num_outputs])
            
            outputs.append(x)

        return outputs
   
    
@DETECTORS.register
class YOLOv5(tf.keras.Model):
    def __init__(self, cfg, return_loss=True, **kwargs):
        super(YOLOv5, self).__init__(**kwargs)
        
        self.cfg = cfg 
        self.return_loss = return_loss 
        self.m, self.save = self._create_model()
        self.nms = build_nms(**cfg.test.as_dict())
        
        self.num_anchors = self.cfg.num_anchors
        self.num_classes = self.cfg.num_classes
        
        self.g = 0.5
        self.off = tf.constant([[0, 0],
                                [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                                # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                                ], tf.float32) * self.g  # offsets
        
        assert "IoU" in cfg.bbox_loss.loss, "Only surpport IoU loss (e.g. IoULoss, GIoU, DIoU, CIoU)."
        self.bbox_loss_func = build_loss(**cfg.bbox_loss.as_dict())
        self.label_loss_func = build_loss(**cfg.label_loss.as_dict())
        self.conf_loss_func = build_loss(**cfg.conf_loss.as_dict())
    
    def load_pretrained_weights(self, pretrained_weights_path=None):
        pass

    @property
    def min_level(self):
        return self.cfg.min_level
    
    @property
    def max_level(self):
        return self.cfg.max_level
    
    def _create_model(self):
        # print("\n%3s%18s%3s  %-40s%-32s" % ("", "from", "n", "module", "arguments"))
        anchors = self.cfg.anchors
        num_anchors = len(anchors[0])
        num_classes = self.cfg.num_classes
        depth_multiple = self.cfg.depth_multiple
        width_multiple = self.cfg.width_multiple
        
        filters = [3]
        final_filters = num_anchors * (num_classes + 5)
        layers = []
        save = []
        for i, (f, n, m, kwargs) in enumerate(self.cfg.model):   # from, number, module, args
            for k, a in kwargs.items():
                try:
                    kwargs[k] = eval(a) if isinstance(a, str) else a
                except:
                    pass
          
            n = max(round(n * depth_multiple), 1) if n > 1 else n  # depth gain
            if m in ["Conv", "BottleneckCSP", "SpatialPyramidPooling", 
                     "DWConv", "MixConv2d", "Focus", "CrossConv", "BottleneckCSP", "C3"]:
                m = eval(m) if isinstance(m, str) else m
                in_filters, out_filters = filters[f], kwargs["filters"]
                out_filters  = (make_divisable(out_filters * width_multiple, 8) 
                                if out_filters != final_filters else out_filters)
                kwargs["filters"] = out_filters
                
                if m is Bottleneck:
                    kwargs["out_filters"] = kwargs["filters"]
                    kwargs["in_filters"] = in_filters
                    kwargs.pop("filters")
                
                if m in [BottleneckCSP, C3]:
                    kwargs["n"] = n
                    n = 1
                    
            elif m == "BatchNorm":
                m = tf.keras.layers.BatchNormalization
            elif m == "Concat":
                m = tf.keras.layers.Concatenate
            elif m == "Upsample":
                m = tf.keras.layers.UpSampling2D
            elif m == "Detect":
                m = eval(m) if isinstance(m, str) else m
                kwargs["strides"] = self.cfg.strides
            else:
                out_filters = filters[f]
            
            if n > 1:
                m_ = tf.keras.layers.Sequential(
                    [m(name=str(j), **kwargs) for j in range(n)], name=str(i))
            else:
                m_ = m(name=str(i), **kwargs)    

            m_.i, m_.f = i, f
            # print("%3s%18s%3s   %-40s" % (i, f, n, str(m)), kwargs)
            save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
            layers.append(m_)
            filters.append(out_filters)

        return layers, sorted(save)

    @tf.function(experimental_relax_shapes=True)
    def call(self, inputs, training=None):
        if self.return_loss:
            inputs, image_info = inputs
        x = tf.image.convert_image_dtype(inputs, tf.float32)
        l_outs = []
        for layer in self.m:
            if layer.f != -1:  # if not from previous layer
                x = (l_outs[layer.f] if isinstance(layer.f, int) 
                     else [x if j == -1 else l_outs[j] for j in layer.f])   # from earlier layers

            x = layer(x, training=training)
                            
            l_outs.append(x if layer.i in self.save else None)
        
        if self.return_loss:
            losses = self.compute_losses(x, image_info)
            return x, losses
        
        return self.get_boxes(x)

    def get_targets_per_level1(self, gt_boxes, gt_labels, grid_shape, anchors):
        with tf.name_scope("get_targets_per_level") :
            bs = tf.shape(gt_boxes)[0]
            gt_boxes = tf.stack([(gt_boxes[..., 1] + gt_boxes[..., 3]) * 0.5,
                                 (gt_boxes[..., 0] + gt_boxes[..., 2]) * 0.5,
                                 gt_boxes[..., 3] - gt_boxes[..., 1],
                                 gt_boxes[..., 2] - gt_boxes[..., 0]], -1)
            targets = tf.concat([gt_boxes, tf.cast(gt_labels[:, :, None], gt_boxes.dtype)], -1)  # [x, y, w, h, l]
            batch_inds = tf.tile(tf.reshape(tf.range(bs), [bs, 1, 1]), [1, tf.shape(gt_boxes)[1], 1])
            targets = tf.concat([tf.cast(batch_inds, targets.dtype), targets], -1)  # [bi, x, y, w, h, l]
            
            targets = tf.reshape(targets, [-1, 6])
            targets = tf.boolean_mask(targets, targets[:, -1] > 0)
            anchor_inds = tf.tile(tf.reshape(tf.range(self.num_anchors), [self.num_anchors, 1, 1]), [1, tf.shape(targets)[0], 1])
            targets = tf.tile(tf.expand_dims(targets, 0), [self.num_anchors, 1, 1])
            targets = tf.concat([targets, tf.cast(anchor_inds, targets.dtype)], -1)
            ratios = targets[..., 3:5] / anchors[:, None]  #[na, num_boxes]
            mask_ratios = tf.reduce_max(tf.maximum(ratios, 1. / ratios), -1) < self.cfg.anchor_threshold
            targets = tf.boolean_mask(targets, mask_ratios)
            
            box_xy = targets[..., 1:3]
            box_xi = tf.cast(grid_shape, box_xy.dtype) - box_xy   # inverse
            # Select the 
            mask_xy = tf.transpose(tf.logical_and(box_xy % 1. < self.g, box_xy > 1.), [1, 0])
            mask_xi = tf.transpose(tf.logical_and(box_xi % 1. < self.g, box_xi > 1.), [1, 0])
            mask = tf.concat([tf.cast(tf.ones([1, tf.shape(mask_xy)[1]]), tf.bool), mask_xy, mask_xi], 0)
            
            offsets = tf.zeros_like(box_xy)[None] + self.off[:, None]
            offsets = tf.boolean_mask(offsets, mask)
            targets = tf.boolean_mask(tf.tile(targets[None], [5, 1, 1]), mask)
            
            outputs = tf.zeros([bs, grid_shape[0], grid_shape[1], self.num_anchors, 5], tf.float32)
            if tf.shape(targets)[0] <= 0:
                return outputs

            bi = tf.cast(targets[:, 0], tf.int32)
            ai = tf.cast(targets[:, -1], tf.int32)
            labels = targets[:, 5:6]
            gij = tf.cast(targets[..., 1:3] - offsets, tf.int32)
            
            indices = tf.stack([bi, gij[:, 1], gij[:, 0], ai], -1)
            t = tf.concat([targets[..., 1:3] - tf.cast(gij, box_xy.dtype), targets[..., 3:5], labels], -1)
                       
            outputs = tf.tensor_scatter_nd_update(outputs, indices, t)
            
            return outputs

    def get_targets_per_level(self, gt_boxes, gt_labels, grid_shape, anchors, strides):
        with tf.name_scope("get_targets_per_level") :
            bs = tf.shape(gt_boxes)[0]
            gt_boxes = tf.stack([gt_boxes[..., 1],
                                 gt_boxes[..., 0],
                                 gt_boxes[..., 3],
                                 gt_boxes[..., 2]], -1)
            targets = tf.concat([gt_boxes, tf.cast(gt_labels[:, :, None], gt_boxes.dtype)], -1)  # [x1, y1, x2, y2, l]
            batch_inds = tf.tile(tf.reshape(tf.range(bs), [bs, 1, 1]), [1, tf.shape(gt_boxes)[1], 1])
            targets = tf.concat([tf.cast(batch_inds, targets.dtype), targets], -1)  # [bi, x1, y1, x2, y2, l]
            
            targets = tf.reshape(targets, [-1, 6])
            targets = tf.boolean_mask(targets, targets[:, -1] > 0)
            anchor_inds = tf.tile(tf.reshape(tf.range(self.num_anchors), [self.num_anchors, 1, 1]), [1, tf.shape(targets)[0], 1])
            targets = tf.tile(tf.expand_dims(targets, 0), [self.num_anchors, 1, 1])
            targets = tf.concat([targets, tf.cast(anchor_inds, targets.dtype)], -1)
            
            # grid_wh = tf.stack([targets[..., 3] - targets[..., 1], targets[..., 4] - targets[..., 2]], -1)
            grid_wh = targets[..., 3:5] - targets[..., 1:3]
            ratios = grid_wh / anchors[:, None]  #[na, num_boxes]
            mask_ratios = tf.reduce_max(tf.maximum(ratios, 1. / ratios), -1) < self.cfg.anchor_threshold
            targets = tf.boolean_mask(targets, mask_ratios)
            
            grid_xy = (targets[..., 1:4] + targets[..., 3:5]) * 0.5
            grid_xy /= strides
            grid_xi = tf.cast(grid_shape, grid_xy.dtype) - grid_xy   # inverse
            # Select the 
            mask_xy = tf.transpose(tf.logical_and(grid_xy % 1. < self.g, grid_xy > 1.), [1, 0])
            mask_xi = tf.transpose(tf.logical_and(grid_xi % 1. < self.g, grid_xi > 1.), [1, 0])
            mask = tf.concat([tf.cast(tf.ones([1, tf.shape(mask_xy)[1]]), tf.bool), mask_xy, mask_xi], 0)
            
            offsets = tf.zeros_like(grid_xy)[None] + self.off[:, None]
            offsets = tf.boolean_mask(offsets, mask)
            targets = tf.boolean_mask(tf.tile(targets[None], [5, 1, 1]), mask)
            grid_xy = tf.boolean_mask(tf.tile(grid_xy[None], [5, 1, 1]), mask)
            
            outputs = tf.zeros([bs, grid_shape[0], grid_shape[1], self.num_anchors, 5], tf.float32)
            if tf.shape(targets)[0] <= 0:
                return outputs

            bi = tf.cast(targets[:, 0], tf.int32)
            ai = tf.cast(targets[:, -1], tf.int32)
            gij = tf.cast(grid_xy - offsets, tf.int32)
            
            indices = tf.stack([bi, gij[:, 1], gij[:, 0], ai], -1)
            
            outputs = tf.tensor_scatter_nd_update(outputs, indices, targets[..., 1:6])
            
            return outputs

    def compute_losses_per_level(self, predictions, targets, anchors, weights, strides):
        with tf.name_scope("compute_losses_per_level"):
            # bs = tf.cast(tf.shape(predictions)[0], tf.float32)
            predictions = tf.cast(predictions, tf.float32)            
            bs = tf.shape(predictions)[0]
            nx, ny = tf.shape(predictions)[2], tf.shape(predictions)[1]
            grid_xy = self.make_grid(nx, ny)
            grid_xy = tf.cast(grid_xy, predictions.dtype)
            grid_xy = tf.tile(grid_xy, [bs, 1, 1, 1, 1])

            anchors = tf.reshape(anchors, shape=[1, 1, 1, self.num_anchors, 2])
            anchors = tf.tile(anchors, [bs, ny, nx, 1, 1])
            pred_xy = (tf.nn.sigmoid(predictions[:, :, :, :, 0:2]) * 2. - 0.5 + grid_xy) * strides
            pred_hw = (tf.nn.sigmoid(predictions[:, :, :, :, 2:4]) * 2) ** 2 * anchors
            # pred_hw = tf.math.exp(predictions[:, :, :, :, 2:4]) * anchors
            pred_boxes = tf.concat([pred_xy - pred_hw * 0.5, pred_xy + pred_hw * 0.5], -1)
            # pred_xy = tf.nn.sigmoid(predictions[..., 0:2]) * 2. - 0.5
            # pred_wh = (tf.nn.sigmoid(predictions[..., 2:4]) * 2.) ** 2. * anchors
            # pred_boxes = tf.concat([pred_xy - pred_wh * 0.5, pred_xy + pred_wh * 0.5], -1)
            
            tgt_boxes = targets[..., 0:4]
            tgt_conf = tf.cast(targets[..., 4:5] > 0, tf.float32)
            
            num_pos = tf.reduce_sum(tgt_conf) + 1e-3
            iou = compute_iou(tgt_boxes, pred_boxes, "ciou")
            
            bbox_loss = tf.reduce_sum((1. - iou) * tf.squeeze(tgt_conf, -1)) / num_pos
            pred_labels = predictions[..., 5:]
            tgt_labels = tf.one_hot(tf.cast(targets[..., 4] - 1., tf.int32), self.num_classes)
            label_loss = self.label_loss_func(tgt_labels, pred_labels, tgt_conf)
            label_loss = tf.reduce_sum(label_loss) / (num_pos * self.num_classes)
            
            pred_conf = predictions[..., 4:5]
            tgt_conf *= ((1.0 - self.cfg.gr) + self.cfg.gr * iou[..., None])
            tgt_conf = tf.stop_gradient(tf.clip_by_value(tgt_conf, 0, 1))
            conf_loss = self.conf_loss_func(tgt_conf, pred_conf)
            conf_loss = tf.reduce_mean(conf_loss)
            
            bs = tf.cast(bs, tf.float32)
            return bbox_loss * bs, label_loss * bs, conf_loss * bs

    def compute_losses(self, predictions, image_info):
        with tf.name_scope("compute_losses"):
            bbox_loss_list = []
            conf_loss_list = []
            label_loss_list = []
            
            total_anchors = tf.convert_to_tensor(self.cfg.anchors, tf.float32)
            for i, level in enumerate(range(self.min_level, self.max_level + 1)):
                grid_shape = tf.shape(predictions[i])[1:3]
                strides = 2 ** level
                anchors = total_anchors[i]
                anchors = tf.reshape(anchors, [self.num_anchors, 2])
                
                targets = self.get_targets_per_level(
                    gt_boxes=tf.cast(image_info["boxes"], tf.float32),
                    gt_labels=tf.cast(image_info["labels"], tf.float32),
                    grid_shape=grid_shape, 
                    anchors=anchors,
                    strides=strides)

                targets = tf.stop_gradient(targets) 
                bbox_loss, label_loss, conf_loss = self.compute_losses_per_level(
                    predictions[i], targets, anchors, self.cfg.balance[i], strides)

                bbox_loss_list.append(bbox_loss)
                conf_loss_list.append(conf_loss)
                label_loss_list.append(label_loss)

            bbox_loss = tf.add_n(bbox_loss_list)
            conf_loss = tf.add_n(conf_loss_list)
            label_loss = tf.add_n(label_loss_list)

            bbox_loss = bbox_loss * self.cfg.box_weight
            label_loss = label_loss * self.cfg.label_weight 
            conf_loss = conf_loss * self.cfg.conf_weight         

            return dict(bbox_loss=bbox_loss, conf_loss=conf_loss, label_loss=label_loss) 
         
    def make_grid(self, width, height):
        with tf.name_scope("make_grid"):
            grid_x, grid_y = tf.meshgrid(tf.range(width), tf.range(height)) 
            grid_xy =tf.stack([grid_x, grid_y], -1)  # (h, w, 2)
            grid_xy = tf.expand_dims(tf.expand_dims(grid_xy, 0), 3)  # (1, h, w, 1, 2)
        
            return grid_xy
        
    def get_boxes(self, predictions):
        with  tf.name_scope("get_boxes"):
            boxes_list = []
            conf_list = []
            label_list = []
            for i, level in enumerate(range(self.min_level, self.max_level + 1)):
                pred = predictions[i]
                bs = tf.shape(pred)[0]
                nx, ny = tf.shape(pred)[2], tf.shape(pred)[1]
                grid_xy = self.make_grid(nx, ny)
                grid_xy = tf.cast(grid_xy, pred.dtype)
                grid_xy = tf.tile(grid_xy, [bs, 1, 1, 1, 1])

                out = tf.sigmoid(pred)
                anchors = tf.constant(self.cfg.anchors[i], shape=[1, 1, 1, self.num_anchors, 2], dtype=out.dtype)
           
                anchors = tf.tile(anchors, [bs, ny, nx, 1, 1])
                strides = 2 ** level
                out_xy = (out[..., 0:2] * 2. - 0.5 + grid_xy) * strides
                out_wh = (out[..., 2:4] * 2.) ** 2. * anchors
                
                out_boxes = tf.stack([out_xy[..., 1] - out_wh[..., 1] * 0.5, 
                                      out_xy[..., 0] - out_wh[..., 0] * 0.5,
                                      out_xy[..., 1] + out_wh[..., 1] * 0.5,
                                      out_xy[..., 0] + out_wh[..., 0] * 0.5], -1)
                
                boxes = tf.reshape(out_boxes, [bs, ny * nx * self.num_anchors, 4])
                conf = tf.reshape(out[..., 4], [bs, ny * nx * self.num_anchors, 1])
                labels = tf.reshape(out[..., 5:], [bs, ny * nx * self.num_anchors, self.num_classes])
                input_size = tf.convert_to_tensor([[ny * strides, nx * strides]], boxes.dtype)
                input_size = tf.tile(input_size, [bs, 1])
                boxes = box_utils.to_normalized_coordinates(
                    boxes, input_size[:, 0:1, None], input_size[:, 1:2, None])
                
                boxes_list.append(boxes)
                conf_list.append(conf)
                label_list.append(labels)
            
            pred_boxes = tf.cast(tf.concat(boxes_list, 1), tf.float32)
            pred_conf = tf.cast(tf.concat(conf_list, 1), tf.float32)
            pred_scores = tf.cast(tf.concat(label_list, 1), tf.float32)   
            
            if "Quality" in self.cfg.test.nms:
                return self.nms(pred_boxes, pred_scores, pred_conf)
            
            pred_scores = pred_scores * pred_conf
                
            return self.nms(pred_boxes, pred_scores)
            

def make_divisable(x, divisor):
    return math.ceil(x / divisor) * divisor


def _load_weight_from_torch(model, torch_weights):
    import torch
    
    torch_model = torch.load(torch_weights, map_location=torch.device("cpu"))
    # for k, v in torch_model["state_dict"].items():
    #     if "tracked" in k:
    #         continue
    #     print(k, v.shape)
        
    for weight in model.weights:
        name = weight.name
        name = name.split(":")[0]
        name = name.replace("/", ".")
        if "batch_norm" in name:
            name = name.replace("batch_norm", "bn")
        if "kernel" in name:
            name = name.replace("kernel", "weight")
        if "gamma" in name:
            name = name.replace("gamma", "weight")
        if "beta" in name:
            name = name.replace("beta", "bias")
        if "moving_mean" in name:
            name = name.replace("moving_mean", "running_mean")
        if "moving_variance" in name:
            name = name.replace("moving_variance", "running_var")
        if "yolo_v5" in name:
            name = name.replace("yolo_v5.", "")
        name = "model." + name
        # print(name, weight.shape)

        tw = torch_model["state_dict"][name].numpy()
        
        if len(tw.shape) == 4:
            tw = np.transpose(tw, (2, 3, 1, 0))
        
        if len(tw.shape) == 2:
            tw = np.transpose(tw, (1, 0))
            
        weight.assign(tw)


if __name__ == "__main__":
    import cv2
    from core import build_optimizer
    from configs.yolov5_config import get_yolov5_config
    from data.datasets.coco_dataset import COCODataset
    from core.metrics.mean_average_precision import mAP
    
    np.set_printoptions(precision=2)

    cfg = get_yolov5_config()
    
    name = "yolov5s"
    # cfg.depth_multiple = 1.33  # 0.33 0.67 1.0 1.33
    # cfg.width_multiple = 1.25  # 0.50 0.75 1.0 1.25
    
    yolov5 = YOLOv5(cfg, return_loss=False)
    gt_boxes = tf.constant([[[0.03819444444444445, 0.6536458333333334, 0.034722222222222224, 0.1440972222222222],
                             [0.18541666666666667, 0.6397569444444444, 0.04027777777777778, 0.1267361111111111],
                             [0.34305555555555556, 0.6831597222222222, 0.044444444444444446, 0.0954861111111111],
                             [0.6583333333333333, 0.3385416666666667, 0.4444444444444444, 0.6736111111111112]]]) * 640
    gt_boxes = tf.stack([gt_boxes[..., 1] - gt_boxes[..., 3] * 0.5,
                         gt_boxes[..., 0] - gt_boxes[..., 2] * 0.5,
                         gt_boxes[..., 1] + gt_boxes[..., 3] * 0.5,
                         gt_boxes[..., 0] + gt_boxes[..., 2] * 0.5], -1)
    gt_labels = tf.constant([[0.0, 0, 0, 2]]) + 1
    
    grid_shapes = [[80, 80], [40, 40], [20, 20]]
    anchors = tf.constant([[[10, 13], [16, 30], [33, 23]], 
                           [[30, 61], [62, 45], [59, 119]], 
                           [[116, 90], [156, 198], [373, 326]]], tf.float32)
    strides = [8, 16, 32]  
        
    outputs = yolov5(tf.cast(tf.random.uniform([1, cfg.train.input_size[0], cfg.train.input_size[1], 3], 0, 255), tf.uint8), 
                     training=False)
    
    # for i in range(3):
    #     targets = yolov5.get_targets_per_level(gt_boxes, gt_labels, grid_shapes[i], anchors[i], strides[i])

    # for w in yolov5.weights:
    #     print(w.name)
    
    coco = COCODataset("/home/bail/Data/data1/Dataset/COCO", 
                       batch_size=1, 
                       augmentations=[
                        #    dict(FlipLeftToRight=dict(probability=0.5)),
                           dict(ResizeV2=dict(short_side=640, long_side=1024, strides=32)),
                        #    dict(Resize=dict(size=(640, 640), strides=32)),
                        ], 
                    #    mosaic=dict(size=(640, 640), min_image_scale=0.25),
                    #    mixup=dict(batch_size=2, alpha=8.0),
                       training=False)
    yolov5.load_weights("/home/bail/Workspace/pretrained_weights/yolov5s.h5")
    metric = mAP(num_classes=80)
    for images, image_info in coco.dataset():
        gt_boxes = image_info["boxes"]
        gt_labels = image_info["labels"]

        outputs = yolov5(images, training=False)
        normalized_factor = tf.cast(tf.tile(tf.expand_dims(image_info["input_size"], 1), [1, 1, 2]), tf.float32)
        pred_boxes = outputs["nmsed_boxes"] * normalized_factor
        pred_scores = outputs["nmsed_scores"]
        pred_classes = outputs["nmsed_classes"]
        metric.update_state(gt_boxes, gt_labels, pred_boxes, pred_scores, pred_classes + 1)

        # img = images[0].numpy()
        # for b in pred_boxes[0].numpy():
        #     c1 = (int(b[1] / 640 * img.shape[1]), int(b[0] / 640 * img.shape[0]))
        #     c2 = (int(b[3] / 640 * img.shape[1]), int(b[2] / 640 * img.shape[0]))
        #     img = cv2.rectangle(img, c1, c2, (0, 0, 255), 1)
        
        # for b in gt_boxes[0].numpy():
        #     c1 = (int(b[1] / 640 * img.shape[1]), int(b[0] / 640 * img.shape[0]))
        #     c2 = (int(b[3] / 640 * img.shape[1]), int(b[2] / 640 * img.shape[0]))
        #     img = cv2.rectangle(img, c1, c2, (255, 0, 0), 1)
        
        # cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # cv2.imshow("image", img)
        # cv2.waitKey(0)

    ap = metric.result()
    print(tf.reduce_mean(ap[:, 0]), tf.reduce_mean(ap))
        # for i in range(3):
        #     img = images[0].numpy().astype(np.uint8)
        #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #     targets = yolov5.get_targets_per_level(gt_boxes, gt_labels, grid_shapes[i], anchors[i], strides[i])
        #     # grid_xy = yolov5.make_grid(tf.shape(targets)[2], tf.shape(targets)[1])
        #     # grid_xy = tf.tile(tf.cast(grid_xy, targets.dtype), [1, 1, 1, 3, 1])
        #     # t_xy = (targets[..., 0:2] + grid_xy) * strides[i]
        #     # t_wh = targets[..., 2:4] * strides[i]

        #     conf = targets[..., 4] > 0
        #     t_box = tf.boolean_mask(targets[..., 0:4], conf)
        #     # t_xy = tf.boolean_mask(t_xy, conf)
        #     # t_wh = tf.boolean_mask(t_wh, conf)
            
        #     # t_box = tf.stack([t_xy[..., 1] - t_wh[..., 1] * 0.5, 
        #     #                   t_xy[..., 0] - t_wh[..., 0] * 0.5,
        #     #                   t_xy[..., 1] + t_wh[..., 1] * 0.5, 
        #     #                   t_xy[..., 0] + t_wh[..., 0] * 0.5], -1)
        #     # t_box = t_box.numpy()

        #     # for b in gt_boxes.numpy()[0]:
        #     #     c1 = (int(b[1] / 640 * img.shape[1]), int(b[0] / 640 * img.shape[0]))
        #     #     c2 = (int(b[3] / 640 * img.shape[1]), int(b[2] / 640 * img.shape[0]))
        #     #     img = cv2.rectangle(img, c1, c2, (0, 255, 0), 1)
                
        #     for b in t_box:
        #         c1 = (int(b[0] / 640 * img.shape[1]), int(b[1] / 640 * img.shape[0]))
        #         c2 = (int(b[2] / 640 * img.shape[1]), int(b[3] / 640 * img.shape[0]))
        #         img = cv2.rectangle(img, c1, c2, (0, 0, 255), 1)
            
        #     cv2.imshow("image", img)
        #     cv2.waitKey(0)
        
        
    # _load_weight_from_torch(yolov5, "/home/bail/Workspace/community_actions/yolov5_model/chouyan.pt")
    
    # yolov5.save_weights("./%s.h5" % name)
    # tf.saved_model.save(yolov5, "./%s" % name)
