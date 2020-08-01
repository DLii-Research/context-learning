import tensorflow as tf
from tensorflow.keras.layers import Layer

import numpy as np

from .utils import hrrs, circ_conv

class Context(Layer):
    
    def __init__(self, contexts=1, atr_model=None, **kwargs):
        super(Context, self).__init__(**kwargs)
        
        # The ATR model handles the switching mechanisms
        self._atr_model = atr_model
        
        # Information Tracking
#         self._context_loss = tf.Variable([0.0, 0.0], name="Context_Losses", trainable=False, dtype=float) # Created in build step
        self._num_contexts = tf.Variable(contexts, name="Num_Contexts", trainable=False, dtype=tf.int64)
        self._hot_context = tf.Variable(0, name="Hot_Context", trainable=False, dtype=tf.int64)
        
        
    def build(self, input_shape):
        # Store the input shape since weights can be rebuilt later
        self._input_shape = int(input_shape[-1])
        
        # Build the ATR model
        self._atr_model.set_context_layer(self)
        self._atr_model.build(self.num_contexts)
        
        # The number of contexts to create in the kernel
        num_kernel_contexts = max(self.num_contexts, self.atr_model.max_num_contexts)
        
        # Create the HRR initializer. This will create the list of HRR vectors
        initializer = lambda shape, dtype=None: hrrs(self._input_shape, n=num_kernel_contexts)
        self.kernel = self.add_weight(name="context", shape=[num_kernel_contexts, self._input_shape], initializer=initializer, trainable=False)
        
        # Store the context losses for each context
        self._context_loss = tf.Variable(np.zeros(num_kernel_contexts), name="Context_Losses", trainable=False, dtype=float)
        
        #TEMP
        self._max_contexts = num_kernel_contexts
        
        
    def call(self, inputs):
        """
        Calculate the output for this layer.
        
        This layer convolves the input values with the context HRR vector
        to produce the output tensor.
        """
        # Fetch the hot context's HRR vector
        context_hrr = self.kernel[self.hot_context]
        
        # Return the resulting convolution between the inputs and the context HRR
        return circ_conv(inputs, context_hrr)
    
    
    def update_and_switch(self, epoch, dynamic_switch, verbose):
        """
        Update ATR values and switch contexts if necessary.
        Returns True if no context switch occurs; False otherwise
        """
        # If there is no ATR model, there's nothing to update
        if self._atr_model is None:
            return True
        
        # Update the ATR madel
        result = self._atr_model.update_and_switch(epoch, self.context_loss, dynamic_switch, verbose)
        
        # Clear the context loss when we're done
        self.clear_context_loss()
        
        # Did the ATR model update or switch?
        return result
        
    
    #TODO Context adding
    def add_context(self):
        # kernel_arr = self.kernel.value()
        # num_hrrs = max(0, self._num_contexts - len(kernel_arr))
        # initializer = lambda shape, dtype=None: np.append(kernel_arr[:self.num_contexts], hrrs(self._input_shape, n=num_hrrs), axis=0)
        # new_weights = self.add_weight(name="context", shape=[self.num_contexts, self._input_shape], initializer=initializer, trainable=False)
        # Create the weights for the layer.
        # The weights in this layer are generated HRR vectors, and are never updated.
        # self.kernel = new_weights
        
        if self._num_contexts < self._max_contexts:
            self._num_contexts.assign_add(1)
            return True
        return False
        
    
    def clear_context_loss(self):
        """Clear the context loss for the current epoch"""
        self._context_loss.scatter_nd_update([[self.hot_context]], [0.0])
    
    
    def add_context_loss(self, context_loss):
        """Accumulate context loss"""
        if self._atr_model is not None:
            context_loss = self._atr_model.context_loss_fn(context_loss)
        else:
            context_loss = tf.keras.losses.mean_squared_error(np.zeros(len(context_loss)), context_loss)
        self._context_loss.scatter_nd_add([[self.hot_context]], [context_loss])
        
        
    def next_context(self):
        """Switch to the next sequential context"""
        self.hot_context = (self.hot_context + 1) % self.num_contexts
        
    
    @property
    def atr_model(self):
        return self._atr_model
    
    @property
    def context_loss(self):
        return self._context_loss[self.hot_context]
    
    @property
    def context_losses(self):
        return self._context_loss
        
    @property
    def num_contexts(self):
        return self._num_contexts.value()
    
    @property
    def hot_context(self):
        """Get the active context index"""
        return self._hot_context.value()
    
    @hot_context.setter
    def hot_context(self, hot_context):
        if hot_context not in range(self.num_contexts):
            raise ValueError("`Provided context does not exist")
        self._hot_context.assign(hot_context)