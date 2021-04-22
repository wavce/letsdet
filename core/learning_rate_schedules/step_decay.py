import tensorflow as tf
from ..builder import LR_SCHEDULERS


@LR_SCHEDULERS.register
class StepDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, decay_steps, decay_rate, name="StepDecay"):
        super(StepDecay, self).__init__(name=name)

        self.lr = initial_learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate

    def __call__(self, global_step):
        with tf.name_scope("StepDecay"):
            self.lr = tf.convert_to_tensor(self.lr, name="initial_learning_rate")
            dtype = self.lr.dtype
            decay_rate = tf.cast(self.decay_rate, dtype)

            if tf.equal(global_step % self.decay_steps, 0):
                self.lr = tf.multiply(self.lr, tf.pow(decay_rate, global_step // self.decay_steps))

            return self.lr

    def get_config(self):
        return {"initial_learning_rate": self.lr,
                "decay_steps": self.decay_steps,
                "decay_rate": self.decay_rate,
                "name": self.name}
