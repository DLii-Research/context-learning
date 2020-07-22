from collections import defaultdict
import numpy as np
import random
import sklearn
import tensorflow as tf
from tensorflow.keras import Model
from .utils import copy_metric, display_progress, plot, plotFrames

from .layer import Context

class NTaskModel(Model):
    def __init__(self, *args, **kwargs):
        super(NTaskModel, self).__init__(*args, **kwargs)
        
        #! A temporary way to store things...
        # These are assigned in `compile`
        self.loss_fn = None
        self.optimizer = None
        self.metric_list = None
        
        # A list of all context layers
        self.context_layers = []
        
        # Debugging
        self.atr_frames = {}
        self.metric_frames = {}
        
        # Misc
        self.total_epochs = 0
        
        # Create the context layer lookup table
        self._create_context_layer_index_list()
        
        
    def compile(self, *args, loss=None, optimizer=None, metrics=[], **kwargs):
        super(NTaskModel, self).compile(*args, **kwargs)
        self._create_context_layer_index_list()
        
        self.loss_fn = loss
        self.optimizer = optimizer
        self.metric_list = [(m, copy_metric(m)) for m in metrics] # A copy of each metric is used for validation data
        
        self.metric_list = {metric.name: [[], []] for metric in metrics} # Training/validation frame sets
        
    
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
        
        return context_delta
        
    
    def _custom_forward_pass(self, x_train, y_train, batch_size):
        """
        This is the training forward pass for an entire epoch

        !!!!! Read through this code as it is a nonstandard training forward pass ( different than model.fit() )
        & NOTE that this does not apply the gradients ie. this does not do a weight update/learn

        """
        
        grads = None
        all_predictions = np.empty((0,) + y_train.shape[1:])
        
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
            
            # Add up the total gradients
            if grads is None:
                grads = gradients
            else:
                grads = np.add(grads, gradients)
                
            # Append the prediction list
#             all_predictions = np.append(all_predictions, predictions, axis=0)
            
            for context_layer_idx in self.context_layers:
                self.layers[context_layer_idx].add_context_loss(self._calc_context_loss(context_layer_idx, gradients))
                
        return all_predictions, grads
                
        
    def fit(self, x_train, y_train, epochs=1, shuffle=True, progress=False, explicit_contexts=None, batch_size=32, validation_data=None, verbose=1):
        
        # Explicit context learning: specify the contexts for ecah of the layers. None=dynamic
        if explicit_contexts is not None:
            if len(explicit_contexts) != len(self.context_layers):
                raise ValueError("Length of explicit contexts does not match the number of context layers")
            for i, idx in enumerate(self.context_layers):
                if explicit_contexts[i] is not None:
                    self.layers[idx].set_hot_context(explicit_contexts[i])
        else:
            explicit_contexts = [None for x in self.context_layers]
        
        # Allocate the memory for plot data
#         self._plot_data_memory_allocate(epochs)
        
        # Determine the default batch size
        if batch_size is None:
            batch_size = len(x_train)
        
        # Shuffle the dataset
        x_train, y_train = sklearn.utils.shuffle(x_train, y_train)
        
        epoch = 0
        while epoch < epochs:
            
            # DEBUG; Display progress
            if progress:
                display_progress(epoch / epochs, title=str([self.layers[i].get_hot_context() for i in self.context_layers]))
            
            # Reset the context loss
            for idx in self.context_layers:
                self.layers[idx].context_loss = 0.0
            
            # Perform a forward pass
            predictions, grads = self._custom_forward_pass(x_train, y_train, batch_size)
            
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
            
            # If no task switch occurred, we can update the weights of the network
            if not switched:
                epoch += 1
                self.total_epochs += 1
                
                # Apply the gradients
                self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
                
                # Update metrics for validation data
#                 self._update_metrics(y_train, predictions, validation_data)
                
                # Add data to plots
                self._add_plot_data()
                        
                # Reshuffle the dataset
                x_train, y_train = sklearn.utils.shuffle(x_train, y_train)
                
                
    def _update_metrics(self, y, y_pred, validation_data):
        """Update any metrics. Also adds data to plots"""
        
        # @TODO
        # Confirmed bug in Keras. compiled_metrics crashes.
        # Forced to use `metrics` with actual class instances
        # This should be the following:
        # 
        # self.compiled_metrics.update_state()
        # 
        # This will be temporarily replaced by:
        for i, (metric_train, metric_test) in enumerate(self.metric_list):
            metric_train.update_state(y, y_pred)
            if validation_data:
                x_test, y_test = validation_data
                metric_train.update_state(y_test, self.predict(x_test))
                
                
    def plot_data_memory_allocate(self, epochs):
        print("Pre-allocating...")
        for group in self.metric_list:
            for i, metric in enumerate(group): # train, test
                self.metric_frames[metric.name][i] += [None]*epochs
            
    
    def _add_plot_data(self):
        
        # Metric values
#         for group in self.metric_list:
#             for i, metric in enumerate(group): # train, test
#                 self.metric_frames[metric.name][i][self.total_epochs-1] = metric.result().numpy()
        
        # ATR values
        for idx in self.context_layers[::-1]:
            for t in range(self.layers[idx].num_tasks):
                self.atr_frames[idx][t][self.total_epochs] = self.layers[idx].atr_model.atr_values[t]
            
        
    def get_contexts(self):
        """Get the hot context from all context layers"""
        return [self.layers[i].get_hot_context() for i in self.context_layers]
    
    
    def set_contexts(self, contexts: list):
        """Set the hot context for the specified layers"""
        if len(contexts) != len(self.context_layers):
            raise ValueError("Provided context list does not match the number of context layers")
        for i, context in enumerate(contexts):
            self.layers[self.context_layers[i]].set_hot_context(context)
            
        
    def plot(self):
        """Plot everything"""
#         self.plot_metrics()
        self.plot_atr_values()
        
        
    def plot_metrics(self):
        scale = 3
        rows, cols = 1, len(self.metric_list)
        fig, *axs = plt.subplots(rows, cols, figsize=(1.5*scale*cols, 1.5*scale*rows))
        fig.suptitle("Metrics")
        for i, ax in enumerate(axs):
            name = self.metric_list[i][0].name
            frames = self.metric_frames[name]
            plotFrames(ax, name, ["Training", "Validation"], *frames, xlabel="Epoch")
            ax.grid()
        plt.legend()
            
        
    def plot_atr_values(self):
        for idx in self.context_layers:
            n = self.layers[idx].num_tasks
            plot(f"ATR Values for Context Layer {idx}", [i for i in range(n)], *self.atr_frames[idx].values())
            
            
            
class NTaskModelBatchedLearning(NTaskModel):
    
    def _custom_forward_pass(self, x_train, y_train, batch_size):
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
            
            # Apply the gradients
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
            
            for context_layer_idx in self.context_layers:
                self.layers[context_layer_idx].add_context_loss(self._calc_context_loss(context_layer_idx, gradients))
                
        
    def fit(self, x_train, y_train, epochs=1, shuffle=True, progress=False, explicit_contexts=None, batch_size=32, validation_data=None, verbose=1):
        
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
        while epoch < epochs:
            
            # DEBUG; Display progress
            if progress:
                display_progress(epoch / epochs, title=str([self.layers[i].get_hot_context() for i in self.context_layers]))
            
            # Reset the context loss
            for idx in self.context_layers:
                self.layers[idx].context_loss = 0.0
                
            # Back up the weights in case we switch
            weights = self.get_weights()
            
            # Perform a forward pass
            self._custom_forward_pass(x_train, y_train, batch_size)
            
            # Iterate backwards over the context layers. If a context switch occurs, don't check any other layers
            switched = False
            for i in range(len(self.context_layers) - 1, -1, -1):
                # Fetch the context layer
                context = self.layers[self.context_layers[i]]
                
                # Check if explicit context learning for this layer is set
                dynamic_switch = explicit_contexts[i] is None
                
                # Update the layer
                update_result = context.update_and_switch(dynamic_switch, verbose=verbose, do_kernel_update=False)
                
                # Update the layer and indicate if a task switch occurred
                if update_result & Context.RESULT_SWITCHED:
                    # A task switch occurred, don't update any other layers/weights
                    switched = True
                    
                    # Restore the weights since this task was invalid
                    self.set_weights(weights)
                    
                    # Update the kernel for the context layer
                    context.update_kernel()
                        
                    # Stop propogation
                    break
                
            # The epoch was a success!
            if not switched:
                
                # Add data to plots
                self._add_plot_data()
                
                epoch += 1
                self.total_epochs += 1
                        
                # Reshuffle the dataset
                x_train, y_train = sklearn.utils.shuffle(x_train, y_train)