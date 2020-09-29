import tensorflow as tf
import numpy as np

from .flags import Verbosity
from .utils import trace

class AtrModel:
    
    def __init__(self, switch_threshold, add_threshold=0.0, max_contexts=0, switch_delay=0):
        """
        switch_threshold: A negative floating point value indicating the threshold to trigger a switch
        add_threshold: A negative floating point value indicating the threshold to add a context
        max_contexts: The maximum number of contexts that can be dynamically allocated
        switch_delay: The number of absorbing epochs to disable autonomous switching for newly initialized contexts
        """
        
        self._switch_threshold = tf.Variable(switch_threshold, name="Switch_Threshold", trainable=False, dtype=tf.float32)
        self._add_threshold = tf.Variable(add_threshold or 0.0, name="Add_Threshold", trainable=False, dtype=tf.float32)
        self._max_contexts = max(max_contexts, 0)
        self._switch_delay = max(switch_delay, 0)
        self._context_layer = None
        
        # Enable the ability to delay switching for a specified number of epochs on newly initialized contexts
        self.delayed_epochs = tf.Variable(self._switch_delay, name="Delayed_Epochs_Counter", trainable=False, dtype=tf.int64)
        
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
        num_contexts = tf.math.maximum(num_contexts, self._max_contexts)
        
        # Create the list of ATR values to track
        self.values = tf.Variable(np.zeros(num_contexts), name="ATR_Values", trainable=False, dtype=tf.float32)
        
        # A second list is created to determine uninitialized ATR values
        self.values_initialized = tf.Variable(np.zeros(num_contexts), name="ATR_Values_Initialized", trainable=False, dtype=tf.bool)
        
    
    def add_context(self):
        # Add the new context to the context layer
        self._context_layer.add_context()
        
        
    def reset_switch_delay(self, verbose=0):
        if self._switch_delay:
            self.delayed_epochs.assign(self._switch_delay)
            if verbose & Verbosity.Contexts:
                tf.print(f"\n[{self.context_layer.name}] Uninitialized context: Context switching disabled for {self._switch_delay} absorbing epochs")
        
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
                    self.reset_switch_delay(verbose)
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
            if not self.values_initialized[self.hot_context]:
                self.reset_switch_delay(verbose)
                
    
    def update_and_switch(self, epoch, context_loss, auto_switch, absorb, retry_fit, verbose):
        """
        Update the ATR.
        
        Returns True if no switch occurred and the ATR values were updated; otherwise False
        """
        if auto_switch and self.delayed_epochs <= 0 and self.should_switch(epoch, context_loss):
            
            # Before we switch...
            self.on_before_switch(epoch, context_loss)
            
            # Count the switches
            self._num_seq_switches.assign_add(1)
            self.epoch_switched.assign(epoch)
            
            # Switch contexts and return the result
            self.switch_contexts(context_loss, verbose)
            
            if (verbose & Verbosity.Contexts) and not retry_fit:
                tf.print(f"\n[{self.context_layer.name}] (no retry) Switched context to {self.hot_context}")
            
            # Switched, so nothing was updated
            return False
        
        # Before the ATR value is updated...
        # This is a sort of hack to skip updating delta
        # traces after a best-fit was found.
        if epoch != self.epoch_switched:
            self.on_before_update(context_loss)
        
        # Update the ATR value
        self.update_atr_value(context_loss, switched=False)
            
        if absorb:
            # Decrement the delayed epochs counter on absorbing states
            if self.delayed_epochs > 0:
                self.delayed_epochs.assign_sub(1)
                
            # Reset the sequence counter on absorbing states
            if self._num_seq_switches != 0:
                self._num_seq_switches.assign(0)
                if (verbose & Verbosity.Contexts) and retry_fit: # If not retry, then this message was already printed above
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
        return True
    
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
                trace("Switch Threshold", self.switch_threshold, '--', 'grey'), # Dark grey is lighter than grey...
                trace("Add Threshold", self.add_threshold, '-.', 'grey', condition=self.max_num_contexts>0),
                trace("Context Delta", self.delta_switched.value(), '-', condition=self.epoch_switched==epoch),
                trace("Context Delta", self.delta.value(), '-')
            ]
        }
    
    # Utility Functions ---------------------------------------------------------------------------
    
    def backup(self):
        """
        Temporarily create a backup of the values
        """
        self._backup = {
            "switch_threshold": self.switch_threshold,
            "add_threshold": self.add_threshold,
            "num_seq_switches": self._num_seq_switches.value(),
            "epoch_switched": self.epoch_switched.value(),
            "delta": self.delta.value(),
            "delta_switched": self.delta_switched.value(),
            "values": self.values.value(),
            "values_initialized": self.values_initialized.value()
        }
        
    def restore(self):
        """
        Restore the values from a temporary backup
        """
        self.switch_threshold = self._backup["switch_threshold"]
        self.add_threshold = self._backup["add_threshold"]
        self._num_seq_switches.assign(self._backup["num_seq_switches"])
        self.epoch_switched.assign(self._backup["epoch_switched"])
        self.delta.assign(self._backup["delta"])
        self.delta_switched.assign(self._backup["delta_switched"])
        self.values.assign(self._backup["values"])
        self.values_initialized.assign(self._backup["values_initialized"])
        
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
        
    @property
    def switch_threshold(self):
        return self._switch_threshold.value()
    
    @switch_threshold.setter
    def switch_threshold(self, threshold):
        self._switch_threshold.assign(threshold)
        
    @property
    def add_threshold(self):
        return self._add_threshold.value()
    
    @add_threshold.setter
    def add_threshold(self, threshold):
        return self._add_threshold.assign(threshold)

    
# Atr Implementations -----------------------------------------------------------------------------

class AtrMovingAverage(AtrModel):
    def update_atr_value(self, context_loss, switched):
        if switched or not self.values_initialized[self.hot_context]:
            self.set_atr_value(context_loss)
        else:
            self.set_atr_value((self.values[self.hot_context] + context_loss) / 2.0)
        return True
            
class TdErrorSwitch(AtrModel):
    def __init__(self, learn_rate, switch_threshold, add_threshold=0.0, max_contexts=0, switch_delay=0):
        super(TdErrorSwitch, self).__init__(switch_threshold, add_threshold, max_contexts, switch_delay)
        self.learn_rate = learn_rate
    
    def update_atr_value(self, context_loss, switched):
        if switched or not self.values_initialized[self.hot_context]:
            self.set_atr_value(context_loss)
        else:
            delta = context_loss - self.values[self.hot_context]
            self.set_atr_value(self.values[self.hot_context] + self.learn_rate*delta)
        return True