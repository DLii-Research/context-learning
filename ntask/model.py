from collections import defaultdict
import numpy as np
import random
import sklearn
import tensorflow as tf
from tensorflow.keras import Model
from .utils import display_progress, plotFrames

from .layer import Context

class NTaskModel(Model):
    def __init__(self, *args, loss_fn=None, optimizer=None, **kwargs):
        super(NTaskModel, self).__init__(*args, **kwargs)
        
        #! A temporary way to store things...
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        
        # A list of all context layers
        self.context_layers = []
        
        # Debugging
        self.atr_frames = {}
        
        # Misc
        self.total_epochs = 0
        
        # Create the context layer lookup table
        self._create_context_layer_index_list()
        
        
    def compile(self, *args, **kwargs):
        super(NTaskModel, self).compile(*args, **kwargs)
        self.create_context_layer_index_list()
        
    
    def _create_context_layer_index_list(self):
        """
        Create a lookup table for context layers.
        This is required to efficiently update each context layer's loss information
        """
        self.context_layers = [i for i, layer in enumerate(self.layers) if isinstance(layer, Context)]
        self.atr_frames = {i: defaultdict(dict) for i in self.context_layers}
        
        
    def _calc_context_loss(self, context_layer_idx, gradients):
        """
        IMPORTANT: 
        1) Assumes no use of activation function on Ntask layer
        2) Assumes that the layer following the Ntask layer:
            a) Is a Dense layer
            b) Is using bias 
               — ex: Dense(20, ... , use_bias=True) 
               — note Keras Dense layer uses bias by default if no value is given for use_bias param
        3) Assumes index of the next layer's gradient is known within the gradients list returned from gradient tape in a tape.gradient call
        4) If the above points aren't met, things will break and it may be hard to locate the bugs
        """
        # From the delta rule in neural network math
        delta_at_next_layer = gradients[context_layer_idx + 1]
        transpose_of_weights_at_next_layer = tf.transpose(self.layers[context_layer_idx + 1].get_weights()[0])
        
        # Calculate delta at n-task layer
        context_delta = np.dot(delta_at_next_layer, transpose_of_weights_at_next_layer).astype(np.float)
        
        # Calculate Context Error
        # Keras MSE must have both args be arrs of floats, if one or both are arrs of ints, the output will be rounded to an int
        # This is how responsible the context layer was for the loss
        return tf.keras.losses.mean_squared_error(np.zeros(len(context_delta)), context_delta)
        
    
    def _custom_forward_pass(self, x_train, y_train, epoch_grads, batch_size):
        """
        This is the training forward pass for an entire epoch

        !!!!! Read through this code as it is a nonstandard training forward pass ( different than model.fit() )
        & NOTE that this does not apply the gradients ie. this does not do a weight update/learn

        """
        
        # Calculate the total number of batches that need to be processed
        num_batches = int(np.ceil(len(x_train) / batch_size))
        
        # Tensorflow 2 style training -- info can be found here: https://www.tensorflow.org/guide/effective_tf2 
        # This is similar to model.fit(), however this is a custom training loop -- ie. it does things differently than model.fit()
        # look at each input and label (there are 4 for the logic gates)
        for start, end in ((s*batch_size, (s + 1)*batch_size) for s in range(num_batches)):
            
            # Slice into batch
            x = x_train[start:end]
            y = y_train[start:end]
            
            with tf.GradientTape() as tape:
                predictions = self(x, training=True) # Forward pass
                loss = self.loss_fn(y, predictions) # Get the loss
            
            # Extract the gradients for the loss of the current sample
            gradients = tape.gradient(loss, self.trainable_variables)
            
            # Collect the gradients from each sample in the dataset for the epoch
            epoch_grads.append(gradients)
            
            for context_layer_idx in self.context_layers:
                self.layers[context_layer_idx].context_loss += self._calc_context_loss(context_layer_idx, gradients)
                
        
    def fit(self, x_train, y_train, n_epochs, shuffle=True, progress=False, explicit_contexts=None, batch_size=None, verbose=1):
        
        # Explicit context learning: specify the contexts for ecah of the layers. None=dynamic
        if explicit_contexts is not None:
            if len(explicit_contexts) != len(self.context_layers):
                raise ValueError("Length of explicit contexts does not match the number of context layers")
            for i, idx in enumerate(self.context_layers):
                if explicit_contexts[i] is not None:
                    self.layers[idx].set_hot_context(explicit_contexts[i])
        else:
            explicit_contexts = [None for x in self.context_layers]
        
        # Determine the default batch size
        if batch_size is None:
            batch_size = len(x_train)
        
        # Shuffle the dataset
        x_train, y_train = sklearn.utils.shuffle(x_train, y_train)
        
        epoch = 0
        while epoch < n_epochs:
            
            # DEBUG; Display progress
            if progress:
                display_progress(epoch / n_epochs, title=str([self.layers[i].get_hot_context() for i in self.context_layers]))
            
            # initialize the values for the loop
            epoch_grads = []
            for idx in self.context_layers:
                self.layers[idx].context_loss = 0.0
            
            # Perform a forward pass
            self._custom_forward_pass(x_train, y_train, epoch_grads, batch_size)
            
            # Iterate backwards over the context layers. If a context switch occurs, don't check any other layers
            switched = False
            for i in range(len(self.context_layers) - 1, -1, -1):
                # Fetch the context layer
                context = self.layers[self.context_layers[i]]
                
                # Check if explicit context learning for this layer is set
                dynamic_switch = explicit_contexts[i] is None
                
                # Update the layer and indicate if a task switch occurred
                if context.update_and_switch(dynamic_switch, verbose=verbose) & Context.RESULT_SWITCHED:
                    # A task switch occurred, don't update any other layers/weights
                    switched = True
                    break
            
            # If no task switch occurred, we can update the weights of the network
            if not switched:
                epoch += 1
                self.total_epochs += 1
                for grads in epoch_grads:
                    self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

                for idx in self.context_layers[::-1]:
                    for t in range(self.layers[idx].num_tasks):
                        self.atr_frames[idx][t][self.total_epochs] = self.layers[idx].atr_model.atr_values[t]
                        
                # Reshuffle the dataset
                x_train, y_train = sklearn.utils.shuffle(x_train, y_train)
                        
                        
    def get_contexts(self):
        """Get the hot context from all context layers"""
        return [self.layers[i].get_hot_context() for i in self.context_layers]
            
                    
    def plot_atr_values(self):
        for idx in self.context_layers:
            n = self.layers[idx].num_tasks
            plotFrames(f"ATR Values for Context Layer {idx}", *self.atr_frames[idx].values(), labels=[i for i in range(n)])