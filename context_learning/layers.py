import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from keras.utils import layer_utils

from .utils import hrr, circular_conv_power

class ContextLayerBase(keras.layers.Layer):
    def __init__(self,
                 switch_threshold=-np.inf,
                 add_threshold=-np.inf,
                 num_contexts=1,
                 max_contexts=None,
                 learn_rate=0.5,
                 expected_loss_init=None,
                 verbose=0):
        super(ContextLayerBase, self).__init__()
        
        self.verbose = verbose
        
        # Indicate the maximum number of contexts that can be allocated
        self.max_contexts = max_contexts
        
        # Track the number of contexts available
        self.num_contexts = tf.Variable(num_contexts, dtype=tf.int32, trainable=False, name="Number of Contexts")
        
        # Contextualization
        self.context = tf.Variable(0, dtype=tf.int32, trainable=False, name="Active Context")
        
        # Switching Mechanism
        self.switch_threshold = tf.Variable(switch_threshold, dtype=tf.float32, trainable=False, name="Switch Threshold")
        self.add_threshold = tf.Variable(add_threshold, dtype=tf.float32, trainable=False, name="Add Threshold")
        self.learn_rate = tf.Variable(learn_rate, dtype=tf.float32, trainable=False, name="Learn Rate")
        self.expected_losses = tf.Variable([np.inf], dtype=tf.float32, trainable=False, name="Expected losses", shape=tf.TensorShape(None))
        self.observed_loss = tf.Variable(0.0, dtype=tf.float32, trainable=False, name="Observed loss")
        self.delta = tf.Variable(0.0, dtype=tf.float32, trainable=False, name="Context Delta")
        
        # Supply an explicit initial loss value
        self.expected_loss_init = lambda: expected_loss_init or self.observed_loss
        
        # Keep track of sequential switches to determine when a new context should be added or best-fit used
        self.num_sequential_switches = tf.Variable(0, dtype=tf.int32, trainable=False, name="Number of Sequencial Switches")
        
        # Track the best-fit context over sequential switches
        self.best_fit_context = tf.Variable(0, dtype=tf.int32, trainable=False, name="Best-fit Context")
        self.best_fit_context_delta = tf.Variable(0.0, dtype=tf.float32, trainable=False, name="Best-fit Context Delta")
        
    def call(self, x, training=None):
        raise NotImplemented()
        
    def count_params(self):
        params = layer_utils.count_params(
            [w for w in self.weights if hasattr(w, "shape") and w.shape.dims != None])
        params += self.num_contexts.numpy() # length of expected_losses tensor
        return params
    
    def add_observed_loss(self, observed_loss):
        loss = self.compute_observed_error(observed_loss)
        self.observed_loss.assign_add(loss)
        
    def update(self, absorb, is_retry):
        
        # Compute the context delta
        self.delta.assign(self.expected_loss() - self.observed_loss)
        
        # Switch or update
        should_switch = self.should_switch()
        tf.cond(
            should_switch,
            self._switch,
            lambda: self._update(absorb, is_retry))
        
        # Reset the observed loss
        self.observed_loss.assign(0.0)
        
        # Return the result of the switch
        return should_switch
    
    # Context Switching ----------------------------------------------------------------------------
    
    def should_switch(self):
        return tf.less_equal(self.delta, self.switch_threshold)
    
    def should_add_context(self):
        # If no add threshold is specified, no adding allowed
        can_add = True if self.max_contexts is None else tf.less(self.num_contexts, self.max_contexts)
        return tf.logical_and(
            can_add,
            tf.less_equal(
                self.best_fit_context_delta,
                self.add_threshold))
    
    def _switch(self):
        if self.verbose >= 2:
            tf.print("\nSwitching contexts...")
        def force_switch():
            if self.verbose >= 2:
                tf.print("Force Switching; context delta:", self.delta, "best fit delta", "")
            tf.cond(
                self.should_add_context(),
                self._switch_to_new_context,
                self._switch_to_best_fit_context)
        def update_best_fit_context():
            self.best_fit_context.assign(self.context)
            self.best_fit_context_delta.assign(self.delta)
        tf.cond(
            tf.logical_or(
                tf.equal(self.num_sequential_switches, 0),
                tf.less(self.delta, self.best_fit_context_delta)),
            update_best_fit_context,
            tf.no_op)
        tf.cond(
            tf.less(self.num_sequential_switches, self.num_contexts - 1),
            self._switch_to_next_context,
            force_switch)
        return True
        
    def _switch_to_next_context(self):
        self.set_context((self.context + 1) % self.num_contexts)
        self.num_sequential_switches.assign_add(1)
        
    def _switch_to_best_fit_context(self):
        if self.verbose >= 1:
            tf.print("Switching to best-fit context:", self.best_fit_context)
        self.set_context(self.best_fit_context)
        self.expected_losses.scatter_nd_update([[self.context]], [self.observed_loss])
    
    def _switch_to_new_context(self):
        if self.verbose >= 1:
            tf.print("\nSwitching to a new context:", self.num_contexts)
        self.add_context()
        self.set_context(self.num_contexts - 1)
    
    # Updating -------------------------------------------------------------------------------------
    
    def _update(self, absorb, is_retry):
        
        # Reset the number of sequential switches on absorbing states
        tf.cond(
            absorb,
            lambda: self.num_sequential_switches.assign(0),
            tf.no_op)
        
        # Update the expected loss
        loss = tf.cond(
            tf.math.is_inf(self.expected_loss()),
            self.expected_loss_init,
            lambda: self.expected_loss() + self.learn_rate*(self.observed_loss - self.expected_loss()))
        self.expected_losses.scatter_nd_update([[self.context]], [loss])
        
        return False # No switch occurred
        
    # Computations ---------------------------------------------------------------------------------
        
    def compute_observed_error(self, observed_loss):
        error = keras.losses.mean_squared_error(tf.zeros_like(observed_loss), observed_loss)
        return error[0]
    
    def expected_loss(self):
        return self.expected_losses[self.context]
        
    # Context Management ---------------------------------------------------------------------------
        
    def add_context(self):
        if self.max_contexts is not None:
            tf.debugging.assert_less(self.num_contexts, self.max_contexts, message="Context limit reached, can't add")
        self.num_contexts.assign_add(1)
        self.expected_losses.assign(tf.concat([self.expected_losses, [np.inf]], axis=0))
        
    def set_context(self, context):
        tf.debugging.assert_non_negative(context, message="Context must be positive")
        tf.debugging.assert_less(context, self.num_contexts, message="Context out of range")
        self.context.assign(context)
        
        
class ContextLayer(ContextLayerBase):
    def __init__(self, *args, **kwargs):
        super(ContextLayer, self).__init__(*args, **kwargs)
        
    def build(self, input_shape):
        super(ContextLayer, self).build(input_shape)
        self.context_hrr = hrr(input_shape[1])
    
    def call(self, x, training=None):
        return circular_conv_power(x, self.context_hrr, self.context)