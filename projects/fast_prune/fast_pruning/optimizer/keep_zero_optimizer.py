
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer


class KeepZeroOptimizer(optimizer.Optimizer):
    def __init__(self,
                 learning_rate: float = 0.01,
                 beta1: float = 0.9,
                 use_locking: bool = False,
                 name: str = "KeepZero"):
        super(KeepZeroOptimizer, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._beta1 = beta1

        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None
        self._beta1_t = None

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._beta1_t = ops.convert_to_tensor(self._beta1, name="beta1")

    def _create_slots(self, var_list):
        first_var = min(var_list, key=lambda x: x.name)
        self._create_non_slot_variable(initial_value=1.0, name='step_counter', colocate_with=first_var)

        # Create slots for the first and second moments.
        for var in var_list:
            self._zeros_slot(var=var, slot_name="m", op_name=self._name)

    def _get_step_counter(self):
        with ops.init_scope():
            graph = ops.get_default_graph()
            return self._get_non_slot_variable('step_counter', graph=graph)

    def _apply_dense(self, grad, var):

        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)

        # maintain momentum
        m = self.get_slot(var, "m")
        m_t = m.assign(beta1_t * m + (1. - beta1_t) * grad)
        new_var = (var - lr_t * m_t)

        mask = tf.where(tf.abs(var) <= 1e-10, tf.fill(var.shape, 0.0), tf.fill(var.shape, 1.0))
        # update var
        var2 = tf.assign(var, new_var * mask)

        return control_flow_ops.group(*[var2, m_t])

    def _resource_apply_dense(self, grad, var):

        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)

        # maintain momentum
        m = self.get_slot(var, "m")
        m_t = m.assign(beta1_t * m + (1. - beta1_t) * grad)
        new_var = (var - lr_t * m_t)

        mask = tf.where(tf.abs(var) <= 1e-10, 0.0, 1.0)
        # update var
        var2 = tf.assign(var, new_var * mask)

        return control_flow_ops.group(*[var2, m_t])

    def _apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")

    def _finish(self, update_ops, name_scope):
        with ops.control_dependencies(update_ops):
            step_counter_t = self._get_step_counter()
            with ops.colocate_with(step_counter_t):
                update_step_counter_t = tf.assign_add(step_counter_t, 1.0, use_locking=self._use_locking)

        return control_flow_ops.group(*update_ops + [update_step_counter_t], name=name_scope)
