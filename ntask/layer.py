from tensorflow.keras.layers import Layer, Input, Dense
import numpy as np

from .utils import hrr, hrrs, circ_conv

class Context(Layer):
    
    RESULT_UPDATED  = 0 # The ATR model was updated successfully
    RESULT_SWITCHED = 1 # A task switch was triggered in the ATR
    RESULT_ADDED    = 2 # A new task was added to the ATR model
    
    def __init__(self, atr_model):
        super(Context, self).__init__()
        
        self.kernel = None
        self.atr_model = atr_model
        
        # Information Tracking
        self.num_tasks = 0
        self.context_loss = 0.0
        
        
    def _setup_hrr_weights(self):
        # Fetch and store the number of tasks. Used to check for dynamically added tasks
        self.num_tasks = self.atr_model.num_tasks

        # Create the HRR initializer. This will create the list of HRR vectors
        if self.kernel is None:
            initializer = lambda shape, dtype=None: hrrs(self._input_shape, n=self.num_tasks)
        else:
            # If there are previously generated HRRs, they should be retained
            kernel_arr = self.kernel.numpy()
            num_hrrs = max(0, self.num_tasks - len(kernel_arr))
            initializer = lambda shape, dtype=None: np.append(kernel_arr[:self.num_tasks], hrrs(self._input_shape, n=num_hrrs), axis=0)
        
        # Create the weights for the layer.
        # The weights in this layer are generated HRR vectors, and are never updated.
        self.kernel = self.add_weight(name="context", shape=[self.num_tasks, self._input_shape], initializer=initializer, trainable=False)
        
    
    def build(self, input_shape):
        
        # Store the input shape since weights can be rebuilt later
        self._input_shape = int(input_shape[-1])
        
        # Build the n-task information
        self._setup_hrr_weights()
        
        
    def call(self, inputs):
        """
        Calculate the output for this layer.
        
        This layer convolves the input values with the context HRR vector
        to produce the output tensor.
        """
        
        # Fetch the hot context's HRR vector
        context_hrr = self.kernel[self.get_hot_context()]
        
        # Return the resulting convolution between the inputs and the context HRR
        return circ_conv(inputs, context_hrr)
        
    
    def get_hot_context(self):
        """Get the active context index"""
        return self.atr_model.hot_context_idx
    
    
    def set_hot_context(self, index):
        self.atr_model.set_hot_context(index)
        self.context_loss = 0.0
    
    
    def next_context(self):
        """Switch to the next sequential context"""
        return self.atr_model.next_context()
    
    
    def update_and_switch(self, dynamic_switch=True, verbose=1):
        """
        Update ATR values and switch contexts if necessary.
        Returns True if no context switch occurs; False otherwise
        """
        # Update the ATR values. If a task switch occurs, check if a task was added...
        if not self.atr_model.update_and_switch(self.context_loss, dynamic_switch, verbose):
            
            # Check if the number of tasks was dynamically manipulated
            if self.num_tasks != self.atr_model.num_tasks:
                
                # Re-initialize the
                self._setup_hrr_weights()
                
                # Determine if a new task was added or removed
                if self.num_tasks < self.atr_model.num_tasks:
                    return Context.RESULT_ADDED | Context.RESULT_SWITCHED
            
            # A task switch occurred, no ATR updates
            return Context.RESULT_SWITCHED
        
        # No task switched occurred, updated successfully
        return Context.RESULT_UPDATED