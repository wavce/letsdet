import tensorflow as tf
from core.bbox.overlaps import compute_iou 
from ..builder import LOSSES


@LOSSES.register
class IoULoss(tf.keras.losses.Loss):
    def __init__(self,
                 eps=1e-4,
                 reduction=tf.losses.Reduction.NONE,
                 weight=1.,
                 box_type="y1x1y2x2",
                 name="IoULoss"):
        super(IoULoss, self).__init__(reduction=reduction, name=name)

        self.eps = eps
        self.weight = weight
        self.box_type = box_type

    def _iou_loss(self, y_true, y_pred, eps):
        """IoU loss.
            Computing the IoU loss between a set of predicted bboxes and target bboxes.
            The loss is calculated as negative log of IoU.
            Args:
                y_pred (Tensor): Predicted bboxes of format (y1, x1, y2, x2),
                    shape (n, 4).
                y_true (Tensor): Corresponding gt bboxes, shape (n, 4).
                eps (float): Eps to avoid log(0).
            Returns:
                Tensor: Loss tensor.
        """
        iou = compute_iou(y_true, y_pred, iou_type="iou")
       
        # loss = -tf.math.log(iou + eps)
        loss = 1. - iou

        weighted_loss = loss * self.weight
                
        return weighted_loss

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        return self._iou_loss(y_true, y_pred, self.eps)


@LOSSES.register
class BoundedIoULoss(tf.keras.losses.Loss):
    def __init__(self, 
                 beta=0.2, 
                 eps=1e-6, 
                 weight=1., 
                 reduction=tf.losses.Reduction.NONE, 
                 name="BoundedIoULoss"):
        super(BoundedIoULoss, self).__init__(reduction=reduction, name=name)

        self.beta = beta
        self.eps = eps
        self.weight = weight

    def _bounded_iou_loss(self, y_true, y_pred, beta=0.2, eps=1e-3):
        """Improving Object Localization with Fitness NMS and Bounded IoU Loss,
           https://arxiv.org/abs/1711.00164.
           Args:
               y_pred (tensor): Predicted bboxes.
               y_true (tensor): Target bboxes.
               beta (float): beta parameter in smoothl1.
               eps (float): eps to avoid NaN.
        """
        py_ctr = (y_pred[:, 0] + y_pred[:, 2]) * 0.5
        px_ctr = (y_pred[:, 2] + y_pred[:, 3]) * 0.5
        ph = y_pred[:, 2] - y_pred[:, 0]
        pw = y_pred[:, 3] - y_pred[:, 1]

        ty_ctr = tf.stop_gradient((y_true[:, 0] + y_true[:, 2]) * 0.5)
        tx_ctr = tf.stop_gradient((y_true[:, 1] + y_true[:, 3]) * 0.5)
        th = tf.stop_gradient(y_true[:, 2] - y_true[:, 0])
        tw = tf.stop_gradient(y_true[:, 3] - y_true[:, 1])

        dx = ty_ctr - py_ctr
        dy = tx_ctr - px_ctr

        dx_loss = 1. - tf.maximum(tf.math.divide_no_nan((tw - 2. * tf.abs(dx)), (tw + 2. * tf.abs(dx))), 0.)
        dy_loss = 1. - tf.maximum(tf.math.divide_no_nan((th - 2. * tf.abs(dy)), (th + 2. * tf.abs(dy))), 0.)
        dw_loss = 1. - tf.minimum(tf.math.divide_no_nan(tw, pw), tf.math.divide_no_nan(pw, tw))
        dh_loss = 1. - tf.minimum(tf.math.divide_no_nan(th, ph), tf.math.divide_no_nan(ph, th))

        loss_comb = tf.stack([dy_loss, dx_loss, dh_loss, dw_loss], axis=-1)

        loss = tf.where(tf.less(loss_comb, beta),
                        0.5 * loss_comb * loss_comb / beta,
                        loss_comb - 0.5 * beta)

        return loss * self.weight

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        return self._bounded_iou_loss(y_true, y_pred, self.beta, self.eps)


@LOSSES.register
class GIoULoss(tf.keras.losses.Loss):
    def __init__(self, 
                 eps=1e-6, 
                 weight=10., 
                 box_type="y1x1y2x2",
                 reduction=tf.losses.Reduction.NONE, name="GIoULoss"):
        super(GIoULoss, self).__init__(reduction, name)

        self.eps = eps
        self.weight = weight
        self.box_type = box_type

    def _giou_loss(self, y_true, y_pred, eps):
        """IoU loss.
            Computing the IoU loss between a set of predicted bboxes and target bboxes.
            The loss is calculated as negative log of IoU.
            Args:
                y_pred (Tensor): Predicted bboxes of format (y1, x1, y2, x2),
                    shape (n, 4).
                y_true (Tensor): Corresponding gt bboxes, shape (n, 4).
                eps (float): Eps to avoid log(0).
            Returns:
                Tensor: Loss tensor.
        """
        if self.box_type == "yxhw":
            y_true = tf.concat([y_true[..., 0:2] - y_true[..., 2:4] * 0.5, y_true[..., 0:2] + y_true[..., 2:4] * 0.5], -1)
            y_pred = tf.concat([y_pred[..., 0:2] - y_pred[..., 2:4] * 0.5, y_pred[..., 0:2] + y_pred[..., 2:4] * 0.5], -1)
        
        giou = compute_iou(y_true, y_pred, iou_type="giou")
        loss = 1. - giou

        weighted_loss = loss * self.weight
                
        return weighted_loss

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        return self._giou_loss(y_true, y_pred, self.eps)


@LOSSES.register
class DIoULoss(tf.keras.losses.Loss):
    def __init__(self, 
                 epsilon=1e-5, 
                 weight=12., 
                 box_type="y1x1y2x2",
                 reduction=tf.losses.Reduction.NONE, 
                 name="DIoULoss"):
        super(DIoULoss, self).__init__(reduction=reduction, name=name)

        self.epsilon = epsilon
        self.weight = weight

        self.box_type = box_type

    def _diou_loss(self, y_true, y_pred):
        if self.box_type == "yxhw":
            y_true = tf.concat([y_true[..., 0:2] - y_true[..., 2:4] * 0.5, y_true[..., 0:2] + y_true[..., 2:4] * 0.5], -1)
            y_pred = tf.concat([y_pred[..., 0:2] - y_pred[..., 2:4] * 0.5, y_pred[..., 0:2] + y_pred[..., 2:4] * 0.5], -1)
        
        diou = compute_iou(y_true, y_pred, iou_type="diou")

        weighted_loss = (1 - diou) * self.weight
                
        return weighted_loss

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        return self._diou_loss(y_true, y_pred)


@LOSSES.register
class CIoULoss(tf.keras.losses.Loss):
    def __init__(self, 
                 epsilon=1e-5, 
                 weight=12., 
                 box_type="y1x1y2x2",
                 reduction=tf.losses.Reduction.NONE, 
                 name="DIoULoss"):
        super(CIoULoss, self).__init__(reduction=reduction, name=name)

        self.epsilon = epsilon
        self.weight = weight
        self.box_type = box_type

    def _ciou_loss(self, y_true, y_pred):
        if self.box_type == "yxhw":
            y_true = tf.concat([y_true[..., 0:2] - y_true[..., 2:4] * 0.5, y_true[..., 0:2] + y_true[..., 2:4] * 0.5], -1)
            y_pred = tf.concat([y_pred[..., 0:2] - y_pred[..., 2:4] * 0.5, y_pred[..., 0:2] + y_pred[..., 2:4] * 0.5], -1)
        
        ciou = compute_iou(y_true, y_pred, iou_type="ciou")
        loss = 1. - ciou

        weighted_loss = loss * self.weight
                
        return weighted_loss
    
    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        return self._ciou_loss(y_true, y_pred)

