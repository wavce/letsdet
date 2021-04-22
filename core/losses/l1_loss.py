import tensorflow as tf
from ..builder import LOSSES


# @LOSSES.register
# class SmoothL1Loss(tf.keras.losses.Loss):
#     def __init__(self, delta=1.0, weight=1., reduction=tf.keras.losses.Reduction.NONE):
#         super(SmoothL1Loss, self).__init__(reduction=reduction, name="SmoothL1Loss")
#         self.weight = weight
#         self.delta = delta

#     def _smooth_l1_loss(self, y_true, y_pred):
#         diff = tf.math.abs(y_pred - y_true)
#         loss = tf.where(diff < self.delta, 0.5 * diff * diff / self.delta, diff - 0.5 * self.delta)  

#         return loss           

#     def call(self, y_true, y_pred):
#         loss = self._smooth_l1_loss(y_true, y_pred)

#         return loss * self.weight

@LOSSES.register
class SmoothL1Loss(tf.keras.losses.Huber):
    def __init__(self, delta=1.0, weight=1., reduction=tf.keras.losses.Reduction.NONE):
        super(SmoothL1Loss, self).__init__(reduction=reduction)
        
        self.weight = weight
        self.delta = delta
    
    def call(self, y_true, y_pred):
        loss = super(SmoothL1Loss, self).call(y_true, y_pred)

        return loss * self.weight


@LOSSES.register
class RegL1Loss(tf.keras.losses.Loss):
    def __init__(self, weight=1., reduction=tf.keras.losses.Reduction.NONE):
        super(RegL1Loss, self).__init__(reduction=reduction)

        self.weight = weight

    def call(self, y_true, y_pred):
        loss = tf.math.abs(y_true - y_pred) * self.weight
       
        return loss
