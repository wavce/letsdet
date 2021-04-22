import tensorflow as tf
from ..builder import LOSSES


@LOSSES.register
class QualityFocalLoss(tf.keras.losses.Loss):
    def __init__(self,
                 from_logits=True,
                 use_sigmoid=True,
                 beta=2.0,
                 reduction=tf.keras.losses.Reduction.SUM,
                 weight=1.,
                 name="QualityFocalLoss"):
        super(QualityFocalLoss, self).__init__(reduction=reduction, name=name)

        assert use_sigmoid, "Only support sigmoid."
        self.use_sigmoid = use_sigmoid
        self.from_logits = from_logits
        self.weight = weight
        self.beta = beta

    def _quality_focal_loss(self, labels, scores, y_pred, beta=2.0, from_logits=True):
        """
        labels (Tensor): Target category label (one-hot) with shape (B, N, C).
        scores (Tensor): target quality label with shape (B, N,).
        y_pred (Tensor): Predicted joint representation of classification
            and quality (IoU) estimation with shape (B, N, C), C is the number of
            classes.
        beta (foat): The beta parameter for calculating the modulating factor.
            Defaults to 2.0.
        from_logits (bool): Is the `y_pred` from logits, default is True.
        """
        with tf.name_scope("quality_focal_loss"):
            pos_mask = labels == 1.
            pos_scores = tf.boolean_mask(scores, tf.reduce_any(pos_mask, -1))
            labels = tf.tensor_scatter_nd_update(labels, tf.where(pos_mask), pos_scores)
            
            if from_logits:
                loss = tf.nn.sigmoid_cross_entropy_with_logits(labels, y_pred)
            else:
                loss = tf.keras.losses.binary_crossentropy(labels, y_pred, False)

            scale_factor = tf.nn.sigmoid(y_pred)
            num_classes = tf.shape(y_pred)[-1]
            pos_scale_factor = tf.abs(tf.tile(tf.expand_dims(scores, -1), [1, 1, num_classes]) - scale_factor)
            scale_factor = tf.where(pos_mask, pos_scale_factor, scale_factor)
            scale_factor = tf.pow(scale_factor, beta) 
            
            weighted_loss = scale_factor * loss * self.weight
            
            return weighted_loss

    def call(self, y_true, y_pred):
        y_true, qulaity_scores = y_true
        return self._quality_focal_loss(y_true, qulaity_scores, y_pred, self.beta, self.from_logits)


@LOSSES.register
class DistributionFocalLoss(tf.keras.losses.Loss):
    """Distribution Focal Loss (DFL) is from `Generalized Focal Loss: Learning
    Qualified and Distributed Bounding Boxes for Dense Object Detection
    <https://arxiv.org/abs/2006.04388>`_.

    Returns:
        torch.Tensor: Loss tensor with shape (N,).
    """
    def __init__(self,
                 from_logits=True,
                 reduction=tf.keras.losses.Reduction.SUM,
                 weight=1.,
                 name="FocalLoss"):
        super(DistributionFocalLoss, self).__init__(reduction=reduction, name=name)

        assert from_logits, "Only support logits."
        self.from_logits = from_logits

        self.weight = weight
    
    def _distribution_focal_loss(self, y_true, y_pred):
        with tf.name_scope("distribution_focal_loss"):
            dist_left = tf.cast(y_true, tf.int64)
            dist_right = dist_left + 1

            weight_left = tf.cast(dist_right, tf.float32) - y_true
            weight_right = y_true - tf.cast(dist_left, tf.float32)

            loss_left = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=dist_left, logits=y_pred) * weight_left
            loss_right = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=dist_right, logits=y_pred) * weight_right

            return loss_left + loss_right
    
    def call(self, y_true, y_pred):
        return self._distribution_focal_loss(y_true, y_pred) * self.weight
