import tensorflow as tf
import numpy as np

from .flags import Verbosity
from .utils import trace

class AtrModel:
    
    def __init__(self, switch_threshold, add_threshold=0.0, max_contexts=0):
        
        self.switch_threshold = tf.Variable(switch_threshold, name="Switch_Threshold", trainable=False, dtype=tf.float32)
        self.add_threshold = tf.Variable(add_threshold or 0.0, name="Add_Threshold", trainable=False, dtype=tf.float32)
        self._max_contexts = max_contexts
        self._context_layer = None
        
        # Track the number of sequential switches so we can determine if no tasks fit the threshold
        self._num_seq_switches = tf.Variable(0, name="Sequential_Switches", trainable=False, dtype=tf.int64)
        
        # Indicate the epoch at which the last switch occurred
        self.epoch_switched = tf.Variable(-1, name="Last_Switch_Epoch", trainable=False, dtype=tf.int64)
        
        # If we tried to add another context after the max number of contexts was reached, we should warn the user
        # and stop checking whether or not a context should be added
        self._exceeded_context_limit = tf.Variable(False, name="Context_Limit_Exceeded", trainable=False, dtype=tf.bool)
        
        # Track the context-loss delta for the active context
        self.delta = tf.Variable(0.0, name="ATR_Delta", trainable=False, dtype=tf.float32)
        
        # Store the delta that triggered the initial context switch
        self.delta_switched = tf.Variable(0.0, name="ATR_Delta_Switched", trainable=False, dtype=tf.float32)
        
        # To be built...
        self.values = None
        self.values_initialized = None
        
        
    def set_context_layer(self, context_layer):
        self._context_layer = context_layer
    
    
    def build(self, num_contexts):
        # Determine the number of contexts to create.
        # Since we can't yet dynamically add contexts, we need
        # to create the list at its max size initially.
        num_contexts = max(num_contexts, self._max_contexts)
        
        # Create the list of ATR values to track
        self.values = tf.Variable(np.zeros(num_contexts), name="ATR_Values", trainable=False, dtype=tf.float32)
        
        # A second list is created to determine uninitialized ATR values
        self.values_initialized = tf.Variable(np.zeros(num_contexts), name="ATR_Values_Initialized", trainable=False, dtype=tf.bool)
        
    
    def add_context(self):
        # Add the new context to the context layer
        self._context_layer.add_context()
        
        
    def switch_contexts(self, context_loss, verbose):
        
        # If we have exhausted the context list, look for the one with the best fit
        if self._num_seq_switches >= self.num_contexts:
            best_fit = self.find_best_fit_context()
            
            # If no context really fits well and we can add more contexts, add a new one
            if self.max_num_contexts > 0 and not self._exceeded_context_limit and self.should_add_context(context_loss, best_fit):
                if self.num_contexts < self.max_num_contexts:
                    self.add_context()
                    self.hot_context = self.num_contexts - 1
                    if verbose & Verbosity.Contexts:
                        tf.print(f"\n[{self.context_layer.name}] Adding context {self.hot_context}")
                else:
                    self._exceeded_context_limit.assign(True)
                    tf.print(f"\n[{self.context_layer.name}] WARNING: Attempted to add context after context limit reached")
                
            # Use the best fit context
            else:
                if verbose & Verbosity.Contexts:
                    tf.print(f"\nUsing best-fit context {best_fit}")
                # Switch to the best-fitting context
                self.hot_context = best_fit
                
                # Before the ATR value is updated...
                self.on_before_update(context_loss)
                
                # Update the ATR value for the new context
                self.update_atr_value(self.context_losses[self.hot_context], switched=True)

        else:
            self.context_layer.next_context()
                
    
    def update_and_switch(self, epoch, context_loss, dynamic_switch, no_retry, verbose):
        """
        Update the ATR.
        
        Returns result type
        """
        if dynamic_switch and self.should_switch(epoch, context_loss):
            
            # Before we switch...
            self.on_before_switch(epoch, context_loss)
            
            # Count the switches
            self._num_seq_switches.assign_add(1)
            self.epoch_switched.assign(epoch)
            
            # Switch contexts and return the result
            self.switch_contexts(context_loss, verbose)
            
            if (verbose & Verbosity.Contexts) and no_retry:
                tf.print(f"\n[{self.context_layer.name}] Switched context to {self.hot_context}")
            
            # Switched, so nothing was updated
            return False
        
        # Before the ATR value is updated...
        # This is a sort of hack to skip updating delta
        # traces after a best-fit was found.
        if epoch != self.epoch_switched:
            self.on_before_update(context_loss)
            
        self.update_atr_value(context_loss, switched=False)
            
        # Reset the switch count if we previously switched
        if self._num_seq_switches != 0:
            self._num_seq_switches.assign(0)
            if (verbose & Verbosity.Contexts) and not no_retry:
                tf.print(f"\n[{self.context_layer.name}] Switched context to {self.hot_context}")
            
        # Updated successfully
        return True
    
    
    def set_atr_value(self, context_loss):
        self.values.scatter_nd_update([[self.hot_context]], [context_loss])
        if not self.values_initialized[self.hot_context]:
            self.values_initialized.scatter_nd_update([[self.hot_context]], [True])

    # Event Handlers ------------------------------------------------------------------------------
    
    def on_begin_train(self):
        self.epoch_switched.assign(-1)
    
    def on_before_switch(self, epoch, context_loss):
        if epoch != self.epoch_switched:
            delta = self.values[self.hot_context] - context_loss
            self.delta_switched.assign(delta)
            self.delta.assign(delta)
            
    def on_before_update(self, context_loss):
        if self.values_initialized[self.hot_context]:
            delta = self.values[self.hot_context] - context_loss
            self.delta.assign(delta)
            
    # Overridable ---------------------------------------------------------------------------------
    
    def context_loss_fn(self, context_delta):
        # Calculate Context Error
        # Keras MSE must have both args be arrs of floats, if one or both are arrs of ints, the output will be rounded to an int
        # This is how responsible the context layer was for the loss
        return tf.keras.losses.mean_squared_error(np.zeros(len(context_delta)), context_delta)
    
    def update_atr_value(self, context_loss, switched):
        """Update the ATR value"""
        # Update the ATR value
        self.set_atr_value(context_loss)
    
    def find_best_fit_context(self):
        """Locate the context index with the best fit"""
        return tf.argmax(tf.subtract(self.values, self.context_losses)[:self.num_contexts])
    
    def should_switch(self, epoch, context_loss):
        # If the ATR value has not been initialized yet, we don't need to switch
        if not self.values_initialized[self.hot_context]:
            return False
        # If the context loss exceeds the threshold
        delta = self.values[self.hot_context] - context_loss
        return delta < self.switch_threshold
    
    def should_add_context(self, context_loss, best_fit_context_idx):
        """
        Determine if a new context should be added
        Note: This is only checked after a switch has been determined
        """
        delta = self.values[self.hot_context] - self.context_losses[best_fit_context_idx]
        return delta < self.add_threshold
    
    def epoch_traces(self, epoch):
        """
        Return a dictionary of traces to plot
        """
        return {
            (None, None, "Context Loss"): [
                trace(f"Context {i}", v) for i, v in enumerate(self.values.value())
                      if self.values_initialized[i] is not None
            ],
            (None, "Epoch", "Context Delta"): [
                trace("Switch Threshold", self.switch_threshold.value(), '--', 'grey'), # Dark grey is lighter than grey...
                trace("Add Threshold", self.add_threshold.value(), '-.', 'grey', condition=self.max_num_contexts>0),
                trace("Context Delta", self.delta_switched.value(), '-', condition=self.epoch_switched==epoch),
                trace("Context Delta", self.delta.value(), '-')
            ]
        }
        
    # Properties ----------------------------------------------------------------------------------
        
    @property
    def context_losses(self):
        return self.context_layer.context_losses
        
    @property
    def num_contexts(self):
        if self.context_layer is None:
            return None
        return self.context_layer.num_contexts
        
    @property
    def max_num_contexts(self):
        return self._max_contexts
    
    @property
    def context_layer(self):
        return self._context_layer
    
    @property
    def hot_context(self):
        return self._context_layer.hot_context
    
    @hot_context.setter
    def hot_context(self, hot_context):
        self._context_layer.hot_context = hot_context

    
# Atr Implementations -----------------------------------------------------------------------------

class AtrMovingAverage(AtrModel):
    def update_atr_value(self, context_loss, switched):
        if switched or not self.values_initialized[self.hot_context]:
            self.set_atr_value(context_loss)
        else:
            self.set_atr_value((self.values[self.hot_context] + context_loss) / 2.0)
            
            
# class AtrDelayedSwitch(AtrMovingAverage):
#     def __init__(self, *args, switch_delay=1, **kwargs):
#         super(AtrDelayedSwitch, self).__init__(*args, **kwargs)
#         self.switch_delay = switch_delay
#         self.epochs_without_switch = 0
        
#     def should_switch(self, context_loss):
#         if self.epochs_without_switch >= self.switch_delay:
#             if super(AtrDelayedSwitch, self).should_switch(context_loss):
#                 self.epochs_without_switch = 0
#                 return True
#         self.epochs_without_switch += 1
#         return False
    
    
# class AtrInitialLoss(AtrMovingAverage):
    
#     def __init__(self, *args, **kwargs):
#         super(AtrInitialLoss, self).__init__(*args, **kwargs)
#         self.initial_atr_value = None
        
#     def update_atr_value(self, context_loss, switched):
#         is_initial = self.get_value() is None
#         super(AtrInitialLoss, self).update_atr_value(context_loss, switched)
#         value = self.get_value()
#         if switched or is_initial:
#             self.initial_atr_value = value
#         elif value < (self.initial_atr_value or value):
#             self.initial_atr_value = None
            
#     def should_switch(self, context_loss):
#         if self.initial_atr_value is None:
#             return super(AtrInitialLoss, self).should_switch(context_loss)
#         return False