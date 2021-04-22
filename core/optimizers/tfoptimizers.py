import tensorflow as tf 
from ..builder import OPTIMIZERS


@OPTIMIZERS.register
class SGD(tf.keras.optimizers.SGD):
    def __init__(self,
                 learning_rate=0.01,
                 momentum=0.0,
                 nesterov=False,
                 name="SGD",
                 **kwargs):
        super(SGD, self).__init__(learning_rate=learning_rate,
                                  momentum=momentum,
                                  nesterov=nesterov,
                                  name=name,
                                  **kwargs)


@OPTIMIZERS.register
class Adadelta(tf.keras.optimizers.Adadelta):
    def __init__(self,
                 learning_rate=0.001,
                 rho=0.95,
                 epsilon=1e-7,
                 name='Adadelta',
                 **kwargs):
        super(Adadelta, self).__init__(learning_rate=learning_rate,
                                       rho=rho,
                                       epsilon=epsilon,
                                       name=name,
                                       **kwargs)


@OPTIMIZERS.register
class Adagrad(tf.keras.optimizers.Adagrad):
    def __init__(self,
                 learning_rate=0.001,
                 initial_accumulator_value=0.1,
                 epsilon=1e-7,
                 name='Adagrad',
                 **kwargs):
        super(Adagrad, self).__init__(learning_rate=learning_rate,
                                      initial_accumulator_value=initial_accumulator_value,
                                      epsilon=epsilon,
                                      name=name,
                                      **kwargs)                           


@OPTIMIZERS.register
class Adam(tf.keras.optimizers.Adam):
    def __init__(self,
                 learning_rate=0.001,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-7,
                 amsgrad=False,
                 name='Adam',
                 **kwargs):
        super(Adam, self).__init__(learning_rate=learning_rate,
                                   beta_1=beta_1,
                                   beta_2=beta_2,
                                   epsilon=epsilon,
                                   amsgrad=amsgrad,
                                   name=name,
                                   **kwargs)


@OPTIMIZERS.register
class RMSprop(tf.keras.optimizers.RMSprop):
    def __init__(self,
                 learning_rate=0.001,
                 rho=0.9,
                 momentum=0.0,
                 epsilon=1e-7,
                 centered=False,
                 name="RMSprop",
                 **kwargs):
        super(RMSprop, self).__init__(learning_rate=learning_rate,
                                      rho=rho,
                                      momentum=momentum,
                                      epsilon=epsilon,
                                      centered=centered,
                                      name=name,
                                      **kwargs)


@OPTIMIZERS.register
class Nadam(tf.keras.optimizers.Nadam):
    def __init__(self,
                 learning_rate=0.001,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-7,
                 name='Nadam',
                 **kwargs):
        super(Nadam, self).__init__(learning_rate=learning_rate,
                                    beta_1=beta_1,
                                    beta_2=beta_2,
                                    epsilon=epsilon,
                                    name=name,
                                    **kwargs)


@OPTIMIZERS.register
class Adamax(tf.keras.optimizers.Adamax):
    def __init__(self,
                 learning_rate=0.001,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-7,
                 name='Adamax',
                 **kwargs):
        super(Adamax, self).__init__(learning_rate=learning_rate,
                                     beta_1=beta_1,
                                     beta_2=beta_2,
                                     epsilon=epsilon,
                                     name=name,
                                     **kwargs)
