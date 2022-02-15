import tensorflow as tf
import tensorflow.keras as keras

from . utils import hrrs, circular_conv

class ContextLayerBase(keras.layers.Layer):
    def __init__(self, switch_threshold, num_contexts=1, atr_size=None, switch_delay=0, auto_switch=True,
                 learn_rate=0.5, switch_init=None, init_mul=1.0, init_loss=None, verbose=0):
        super(ContextLayerBase, self).__init__(self)
        
        self.switch_init = switch_init
        self.init_mul = init_mul
        self.init_loss = init_loss
        self.learn_rate = learn_rate
        self.verbose = verbose
        
        # Contextualization
        self.num_contexts = num_contexts
        self.atr_size = atr_size
        self.switch_delay = switch_delay
        self.project = None
        self.hot_context = tf.Variable(0, name="Hot Context", trainable=False, dtype=tf.int32)
        
        # Switching mechanism
        self.auto_switch = auto_switch
        self.context_losses = tf.Variable(tf.zeros(num_contexts), dtype=tf.float32, trainable=False, name="Context Losses")
        self.switch_threshold = tf.Variable(switch_threshold, dtype=tf.float32, trainable=False, name="Switch Threshold")
        self.delayed_epochs = tf.Variable(switch_delay, dtype=tf.int32, trainable=False, name="Delayed Epochs")
        self.epoch_switched = tf.Variable(-1, dtype=tf.int32, trainable=False, name="Epoch Switched")
        self.num_seq_switches = tf.Variable(0, dtype=tf.int32, trainable=False, name="Sequential Switches")
        self.delta = tf.Variable(0.0, dtype=tf.float32, trainable=False, name="Loss Delta")
        self.delta_switched = tf.Variable(0.0, dtype=tf.float32, trainable=False, name="Delta Switched")
        
        # Track observed losses for each context (used to compute deltas and find best-fit contexts)
        self.observed_losses = tf.Variable(tf.zeros(num_contexts), dtype=tf.float32, trainable=False, name="Observed Loss Values")
        self.losses_initialized = tf.Variable(tf.zeros(num_contexts, dtype=tf.int8), dtype=tf.int8, trainable=False, name="Initialized Loss Values") # bool doesn't work yet...
        
        
    def build(self, input_shape):
        if self.atr_size is not None:
            self.project = keras.layers.Dense(self.atr_size)
        else:
            self.atr_size = input_shape[-1]
        self.atrs = hrrs(self.atr_size, self.num_contexts)
    
    def call(self, x):
        if self.project is not None:
            x = self.project(x)
        atr = tf.gather(self.atrs, self.hot_context)
        return circular_conv(atr, x)
    
    def update_and_switch(self, epoch, absorb, to_context=-1):
        # If no autoswitching, explicit switching is considered...
        if not self.auto_switch:
            to_context = tf.cond(to_context == -1, lambda: self.hot_context, lambda: to_context)
        else:
            # next-in-line context
            to_context = (self.hot_context + 1) % self.num_contexts
        return tf.cond(self.should_switch(epoch, to_context), lambda: self._switch(epoch, absorb, to_context), lambda: self._update(epoch, absorb))
        
    def should_switch(self, epoch, to_context):
        if not self.auto_switch:
            return tf.not_equal(self.hot_context, to_context)
        return tf.logical_and(
            self.delayed_epochs <= 0,
            tf.logical_and(
                self.losses_initialized[self.hot_context] == 1,
                self.compute_context_delta() < self.switch_threshold
            ))
    
    def _switch(self, epoch, absorb, to_context):
        
        tf.print("Switch to", to_context)

        # Before the switch...
        self.on_before_switch(epoch)
        
        # Count the switches
        self.num_seq_switches.assign_add(1)
        self.epoch_switched.assign(epoch)
        
        # Switch contexts and return the result
        self.switch_contexts(absorb, to_context)
        
        # Always return true to indicate a switch occurred
        return True
        
    def switch_contexts(self, absorb, to_context):
        def switch_to_best_fit():
            best_fit = self.find_best_fit_context()
            self.set_context(best_fit)
            self.update_context_loss()
        
        def switch_to_context():
            self.set_context(to_context)
            tf.cond(not self.losses_initialized[self.hot_context], self.reset_switch_delay, tf.no_op)
        
        if not self.auto_switch:
            switch_to_context()
            return
            
        # If we've tried all contexts and none fits well, use the best-fit. Otherwise, use the next context
        tf.cond(self.num_seq_switches >= self.num_contexts, switch_to_best_fit, switch_to_context)
    
    def _update(self, epoch, absorb):
        def update_counters():
            # Decrement the delayed epochs counter on absorbing states
            tf.cond(self.delayed_epochs > 0, lambda: self.delayed_epochs.assign_sub(1), tf.no_op)
            # Reset the sequence counter on absorbing states
            tf.cond(self.num_seq_switches != 0, lambda: self.num_seq_switches.assign(0), tf.no_op)
            
        # Before the context loss value is updated...
        # This is a sort of hack to skip updating delta
        # traces after a best-fit was found
        tf.cond(epoch != self.epoch_switched, self.on_before_update, tf.no_op)
        
        # Update the context-loss value
        self.update_context_loss(self.observed_loss(), absorb)
        
        # If obsorbing, update counters
        tf.cond(absorb, update_counters, tf.no_op)
        return False # Always return false to indicate no switch
    
    def observed_loss(self):
        return self.observed_losses[self.hot_context]
    
    def context_loss(self):
        return self.context_losses[self.hot_context]
    
    def compute_context_delta(self):
        return self.context_loss() - self.observed_loss()
    
    def add_observed_loss(self, observed_loss):
        loss = self.compute_observed_error(observed_loss)
        self.observed_losses.scatter_nd_add([[self.hot_context]], loss)
        
    def clear_observed_loss(self):
        self.observed_losses.scatter_nd_update([[self.hot_context]], [0.0])
                
    def compute_observed_error(self, observed_loss):
        return keras.losses.mean_squared_error(tf.zeros_like(observed_loss), observed_loss)
    
    def set_context_loss(self, context_loss, absorb):
        self.context_losses.scatter_nd_update([[self.hot_context]], [context_loss])
        tf.cond(not self.losses_initialized[self.hot_context] and absorb, lambda: self.losses_initialized.scatter_nd_update([[self.hot_context]], [1]), tf.no_op)
    
    def update_context_loss(self, context_loss, absorb, switched=False):
        def init_context_loss():
            init_loss = self.init_mul*context_loss
            if self.init_loss is not None:
                init_loss = tf.maximum(init_loss, self.init_loss)
            # tf.print("Initializing context loss...")
            self.set_context_loss(init_loss, absorb)
            
        def update_context_loss():
            direct_loss = lambda: context_loss
            delta_loss = lambda: self.context_loss() + self.learn_rate*(self.observed_loss() - self.context_loss())
            loss = tf.cond(switched, direct_loss, delta_loss)
            # tf.print("Updating context loss...")
            self.set_context_loss(loss, absorb)
        
        tf.cond(not self.losses_initialized[self.hot_context], init_context_loss, update_context_loss)
    
    def next_context(self):
        self.hot_context.assign((self.hot_context + 1) % self.num_contexts)
        
    def set_context(self, context):
        self.hot_context.assign(context)
        
    def find_best_fit_context(self):
        return tf.argmax((self.losses - self.context_losses)[:self.num_contexts])
    
    def on_before_switch(self, epoch):
        def update_delta():
            delta = self.compute_context_delta()
            self.delta_switched.assign(delta)
            self.delta.assign(delta)
        tf.cond(epoch != self.epoch_switched, update_delta, tf.no_op)
    
    def on_before_update(self):
        def update_delta():
            delta = self.compute_context_delta()
            self.delta.assign(delta)
        tf.cond(self.losses_initialized[self.hot_context], update_delta, tf.no_op)
        
    def reset_switch_delay(self):
        if self.switch_delay == 0:
            return
        self.delayed_epochs.assign(self.switch_delay)
        if self.verbose:
            tf.print(f"\n[{self.name}] Unitialized context: Context switching disabled for {self._switch_delay} absorbing epochs.")
        
    
class ContextLayer(ContextLayerBase):
    pass