from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.training import training_ops
from tensorflow.python.keras import backend_config
from tensorflow.python.ops import state_ops
from ..builder import OPTIMIZERS


def _gradient_centeralization(grad_and_vars):
    results = []
    for grad, var in grad_and_vars:
        if array_ops.rank(grad) == 4:
            grad -= math_ops.reduce_mean(grad)
        
        results.append((grad, var))

    return results


@OPTIMIZERS.register
class SGDGC(optimizer_v2.OptimizerV2):
    _HAS_AGGREGATE_GRAD = True

    def __init__(self,
                learning_rate=0.01,
                momentum=0.0,
                nesterov=False,
                name="SGDGC",
                **kwargs):
        """Construct a new Stochastic Gradient Descent or Momentum optimizer.

        Arguments:
        learning_rate: A `Tensor`, floating point value, or a schedule that is a
            `tf.keras.optimizers.schedules.LearningRateSchedule`, or a callable
            that takes no arguments and returns the actual value to use. The
            learning rate. Defaults to 0.01.
        momentum: float hyperparameter >= 0 that accelerates SGD in the relevant
            direction and dampens oscillations. Defaults to 0.0, i.e., SGD.
        nesterov: boolean. Whether to apply Nesterov momentum.
            Defaults to `False`.
        name: Optional name prefix for the operations created when applying
            gradients.  Defaults to 'SGD'.
        **kwargs: keyword arguments. Allowed to be {`clipnorm`, `clipvalue`, `lr`,
            `decay`}. `clipnorm` is clip gradients by norm; `clipvalue` is clip
            gradients by value, `decay` is included for backward compatibility to
            allow time inverse decay of learning rate. `lr` is included for backward
            compatibility, recommended to use `learning_rate` instead.
        """
        super(SGDGC, self).__init__(name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._set_hyper("decay", self._initial_decay)

        self._momentum = False
        if isinstance(momentum, ops.Tensor) or callable(momentum) or momentum > 0:
            self._momentum = True
        if isinstance(momentum, (int, float)) and (momentum < 0 or momentum > 1):
            raise ValueError("`momentum` must be between [0, 1].")
        self._set_hyper("momentum", momentum)

        self.nesterov = nesterov

    def _create_slots(self, var_list):
        if self._momentum:
            for var in var_list:
                self.add_slot(var, "momentum")

    def apply_gradients(self,
                        grads_and_vars,
                        name=None,
                        experimental_aggregate_gradients=True):
        """Apply gradients to variables.

        This is the second part of `minimize()`. It returns an `Operation` that
        applies gradients.

        The method sums gradients from all replicas in the presence of
        `tf.distribute.Strategy` by default. You can aggregate gradients yourself by
        passing `experimental_aggregate_gradients=False`.

        Example:

        ```python
        grads = tape.gradient(loss, vars)
        grads = tf.distribute.get_replica_context().all_reduce('sum', grads)
        # Processing aggregated gradients.
        optimizer.apply_gradients(zip(grads, vars),
            experimental_aggregate_gradients=False)

        ```

        Args:
        grads_and_vars: List of (gradient, variable) pairs.
        name: Optional name for the returned operation. Default to the name passed
            to the `Optimizer` constructor.
        experimental_aggregate_gradients: Whether to sum gradients from different
            replicas in the presense of `tf.distribute.Strategy`. If False, it's
            user responsibility to aggregate the gradients. Default to True.

        Returns:
        An `Operation` that applies the specified gradients. The `iterations`
        will be automatically increased by 1.

        Raises:
        TypeError: If `grads_and_vars` is malformed.
        ValueError: If none of the variables have gradients.
        """
        grads_and_vars =  _gradient_centeralization(grad_and_vars=grads_and_vars)
        return super(SGDGC, self).apply_gradients(grads_and_vars=grads_and_vars,
                                                  name=name,
                                                  experimental_aggregate_gradients=experimental_aggregate_gradients)

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(SGDGC, self)._prepare_local(var_device, var_dtype, apply_state)
        apply_state[(var_device, var_dtype)]["momentum"] = array_ops.identity(
            self._get_hyper("momentum", var_dtype))

    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype))
                        or self._fallback_apply_state(var_device, var_dtype))

        if self._momentum:
            momentum_var = self.get_slot(var, "momentum")
            return training_ops.resource_apply_keras_momentum(
                var.handle,
                momentum_var.handle,
                coefficients["lr_t"],
                grad,
                coefficients["momentum"],
                use_locking=self._use_locking,
                use_nesterov=self.nesterov)
        else:
            return training_ops.resource_apply_gradient_descent(
                var.handle, coefficients["lr_t"], grad, use_locking=self._use_locking)

    def _resource_apply_sparse_duplicate_indices(self, grad, var, indices,
                                                **kwargs):
        if self._momentum:
            return super(SGDGC, self)._resource_apply_sparse_duplicate_indices(
                grad, var, indices, **kwargs)
        else:
            var_device, var_dtype = var.device, var.dtype.base_dtype
            coefficients = (kwargs.get("apply_state", {}).get((var_device, var_dtype))
                            or self._fallback_apply_state(var_device, var_dtype))

            return resource_variable_ops.resource_scatter_add(
                var.handle, indices, -grad * coefficients["lr_t"])

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        # This method is only needed for momentum optimization.
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype))
                        or self._fallback_apply_state(var_device, var_dtype))

        momentum_var = self.get_slot(var, "momentum")
        return training_ops.resource_sparse_apply_keras_momentum(
            var.handle,
            momentum_var.handle,
            coefficients["lr_t"],
            grad,
            indices,
            coefficients["momentum"],
            use_locking=self._use_locking,
            use_nesterov=self.nesterov)

    def get_config(self):
        config = super(SGDGC, self).get_config()
        config.update({
            "learning_rate": self._serialize_hyperparameter("learning_rate"),
            "decay": self._serialize_hyperparameter("decay"),
            "momentum": self._serialize_hyperparameter("momentum"),
            "nesterov": self.nesterov,
        })
        return config


@OPTIMIZERS.register
class AdamGC(optimizer_v2.OptimizerV2):
  
    _HAS_AGGREGATE_GRAD = True

    def __init__(self,
                learning_rate=0.001,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-7,
                amsgrad=False,
                name='AdamGC',
                **kwargs):
        super(AdamGC, self).__init__(name, **kwargs)
        self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
        self._set_hyper('decay', self._initial_decay)
        self._set_hyper('beta_1', beta_1)
        self._set_hyper('beta_2', beta_2)
        self.epsilon = epsilon or backend_config.epsilon()
        self.amsgrad = amsgrad

    def _create_slots(self, var_list):
        # Create slots for the first and second moments.
        # Separate for-loops to respect the ordering of slot variables from v1.
        for var in var_list:
            self.add_slot(var, 'm')
        for var in var_list:
            self.add_slot(var, 'v')
        if self.amsgrad:
            for var in var_list:
                self.add_slot(var, 'vhat')

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(AdamGC, self)._prepare_local(var_device, var_dtype, apply_state)

        local_step = math_ops.cast(self.iterations + 1, var_dtype)
        beta_1_t = array_ops.identity(self._get_hyper('beta_1', var_dtype))
        beta_2_t = array_ops.identity(self._get_hyper('beta_2', var_dtype))
        beta_1_power = math_ops.pow(beta_1_t, local_step)
        beta_2_power = math_ops.pow(beta_2_t, local_step)
        lr = (apply_state[(var_device, var_dtype)]['lr_t'] *
            (math_ops.sqrt(1 - beta_2_power) / (1 - beta_1_power)))
        apply_state[(var_device, var_dtype)].update(
            dict(
                lr=lr,
                epsilon=ops.convert_to_tensor_v2(self.epsilon, var_dtype),
                beta_1_t=beta_1_t,
                beta_1_power=beta_1_power,
                one_minus_beta_1_t=1 - beta_1_t,
                beta_2_t=beta_2_t,
                beta_2_power=beta_2_power,
                one_minus_beta_2_t=1 - beta_2_t))

    def set_weights(self, weights):
        params = self.weights
        # If the weights are generated by Keras V1 optimizer, it includes vhats
        # even without amsgrad, i.e, V1 optimizer has 3x + 1 variables, while V2
        # optimizer has 2x + 1 variables. Filter vhats out for compatibility.
        num_vars = int((len(params) - 1) / 2)
        if len(weights) == 3 * num_vars + 1:
            weights = weights[:len(params)]
        super(AdamGC, self).set_weights(weights)
    
    def apply_gradients(self,
                        grads_and_vars,
                        name=None,
                        experimental_aggregate_gradients=True):
        """Apply gradients to variables.

        This is the second part of `minimize()`. It returns an `Operation` that
        applies gradients.

        The method sums gradients from all replicas in the presence of
        `tf.distribute.Strategy` by default. You can aggregate gradients yourself by
        passing `experimental_aggregate_gradients=False`.

        Example:

        ```python
        grads = tape.gradient(loss, vars)
        grads = tf.distribute.get_replica_context().all_reduce('sum', grads)
        # Processing aggregated gradients.
        optimizer.apply_gradients(zip(grads, vars),
            experimental_aggregate_gradients=False)

        ```

        Args:
        grads_and_vars: List of (gradient, variable) pairs.
        name: Optional name for the returned operation. Default to the name passed
            to the `Optimizer` constructor.
        experimental_aggregate_gradients: Whether to sum gradients from different
            replicas in the presense of `tf.distribute.Strategy`. If False, it's
            user responsibility to aggregate the gradients. Default to True.

        Returns:
        An `Operation` that applies the specified gradients. The `iterations`
        will be automatically increased by 1.

        Raises:
        TypeError: If `grads_and_vars` is malformed.
        ValueError: If none of the variables have gradients.
        """
        grads_and_vars =  _gradient_centeralization(grad_and_vars=grads_and_vars)

        return super(AdamGC, self).apply_gradients(grads_and_vars=grads_and_vars,
                                                   name=name,
                                                   experimental_aggregate_gradients=experimental_aggregate_gradients)   

    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype))
                        or self._fallback_apply_state(var_device, var_dtype))

        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')

        if not self.amsgrad:
            return training_ops.resource_apply_adam(
                var.handle,
                m.handle,
                v.handle,
                coefficients['beta_1_power'],
                coefficients['beta_2_power'],
                coefficients['lr_t'],
                coefficients['beta_1_t'],
                coefficients['beta_2_t'],
                coefficients['epsilon'],
                grad,
                use_locking=self._use_locking)
        else:
            vhat = self.get_slot(var, 'vhat')
            return training_ops.resource_apply_adam_with_amsgrad(
                var.handle,
                m.handle,
                v.handle,
                vhat.handle,
                coefficients['beta_1_power'],
                coefficients['beta_2_power'],
                coefficients['lr_t'],
                coefficients['beta_1_t'],
                coefficients['beta_2_t'],
                coefficients['epsilon'],
                grad,
                use_locking=self._use_locking)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype))
                        or self._fallback_apply_state(var_device, var_dtype))

        # m_t = beta1 * m + (1 - beta1) * g_t
        m = self.get_slot(var, 'm')
        m_scaled_g_values = grad * coefficients['one_minus_beta_1_t']
        m_t = state_ops.assign(m, m * coefficients['beta_1_t'],
                               use_locking=self._use_locking)
        with ops.control_dependencies([m_t]):
            m_t = self._resource_scatter_add(m, indices, m_scaled_g_values)

        # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
        v = self.get_slot(var, 'v')
        v_scaled_g_values = (grad * grad) * coefficients['one_minus_beta_2_t']
        v_t = state_ops.assign(v, v * coefficients['beta_2_t'],
                               use_locking=self._use_locking)
        with ops.control_dependencies([v_t]):
            v_t = self._resource_scatter_add(v, indices, v_scaled_g_values)

        if not self.amsgrad:
            v_sqrt = math_ops.sqrt(v_t)
            var_update = state_ops.assign_sub(
                var, coefficients['lr'] * m_t / (v_sqrt + coefficients['epsilon']),
                use_locking=self._use_locking)
            return control_flow_ops.group(*[var_update, m_t, v_t])
        else:
            v_hat = self.get_slot(var, 'vhat')
            v_hat_t = math_ops.maximum(v_hat, v_t)
            with ops.control_dependencies([v_hat_t]):
                v_hat_t = state_ops.assign(
                    v_hat, v_hat_t, use_locking=self._use_locking)
            v_hat_sqrt = math_ops.sqrt(v_hat_t)
            var_update = state_ops.assign_sub(
                var,
                coefficients['lr'] * m_t / (v_hat_sqrt + coefficients['epsilon']),
                use_locking=self._use_locking)
            return control_flow_ops.group(*[var_update, m_t, v_t, v_hat_t])

    def get_config(self):
        config = super(AdamGC, self).get_config()
        config.update({
            'learning_rate': self._serialize_hyperparameter('learning_rate'),
            'decay': self._serialize_hyperparameter('decay'),
            'beta_1': self._serialize_hyperparameter('beta_1'),
            'beta_2': self._serialize_hyperparameter('beta_2'),
            'epsilon': self.epsilon,
            'amsgrad': self.amsgrad,
        })
        return config

