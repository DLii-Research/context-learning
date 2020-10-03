from collections import deque
import copy
import tensorflow as tf

from tensorflow.keras import Model

from tensorflow.python.distribute import parameter_server_strategy
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.keras import backend
from tensorflow.python.keras import callbacks as callbacks_module
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras.mixed_precision.experimental import loss_scale_optimizer as lso
from tensorflow.python.keras.utils import version_utils
from tensorflow.python.profiler import traceme

from .data_handlers import WindowedDataHandler
from .flags import Verbosity
from .layers import Context

# Borrowed from https://github.com/tensorflow/tensorflow/blob/v2.2.0/tensorflow/python/keras/engine/data_adapter.py
try:
    import pandas as pd  # pylint: disable=g-import-not-at-top
except ImportError:
    pd = None

def _minimize(strategy, tape, optimizer, loss, trainable_variables):
    """Minimizes loss for one step by updating `trainable_variables`.
    This is roughly equivalent to
    ```python
    gradients = tape.gradient(loss, trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, trainable_variables))
    ```
    However, this function also applies gradient clipping and loss scaling if the
    optimizer is a LossScaleOptimizer.
    Args:
      strategy: `tf.distribute.Strategy`.
      tape: A gradient tape. The loss must have been computed under this tape.
      optimizer: The optimizer used to minimize the loss.
      loss: The loss tensor.
      trainable_variables: The variables that will be updated in order to minimize
        the loss.
    Return:
      gradients
    """

    with tape:
        if isinstance(optimizer, lso.LossScaleOptimizer):
            loss = optimizer.get_scaled_loss(loss)

    gradients = tape.gradient(loss, trainable_variables)

    # Whether to aggregate gradients outside of optimizer. This requires support
    # of the optimizer and doesn't work with ParameterServerStrategy and
    # CentralStroageStrategy.
    aggregate_grads_outside_optimizer = (
        optimizer._HAS_AGGREGATE_GRAD and  # pylint: disable=protected-access
        not isinstance(strategy.extended,
                       parameter_server_strategy.ParameterServerStrategyExtended))

    if aggregate_grads_outside_optimizer:
        # We aggregate gradients before unscaling them, in case a subclass of
        # LossScaleOptimizer all-reduces in fp16. All-reducing in fp16 can only be
        # done on scaled gradients, not unscaled gradients, for numeric stability.
        gradients = optimizer._aggregate_gradients(zip(gradients,  # pylint: disable=protected-access
                                                       trainable_variables))
    if isinstance(optimizer, lso.LossScaleOptimizer):
        gradients = optimizer.get_unscaled_gradients(gradients)
    gradients = optimizer._clip_gradients(gradients)  # pylint: disable=protected-access
    if trainable_variables:
        if aggregate_grads_outside_optimizer:
            optimizer.apply_gradients(
                zip(gradients, trainable_variables),
                experimental_aggregate_gradients=False)
        else:
            optimizer.apply_gradients(zip(gradients, trainable_variables))
    return gradients
    
# Extended from https://github.com/tensorflow/tensorflow/blob/v2.2.0/tensorflow/python/keras/engine/training.py
class ContextModelBase(Model):
    """
    This abstract model integrates the raw mechanisms and handlers into
    Tensorflow Keras' model class. These mechanisms can be implemented by
    inheriting from this class.
    """
        
    def train_step(self, data):
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        with backprop.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, sample_weight, regularization_losses=self.losses)
            
        gradients = _minimize(self.distribute_strategy, tape, self.optimizer, loss,
              self.trainable_variables)
        
        # Add context loss to layers
        self.add_context_loss(gradients)

        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        return {m.name: m.result() for m in self.metrics}
    
    
    @training.enable_multi_worker
    def fit(self,
            x=None,
            y=None,
            batch_size=None,
            epochs=1,
            verbose=1,
            auto_switch=True,
            retry_fit=True,
            absorb=True,
            train_after_switch=True,
            callbacks=None,
            validation_split=0.,
            validation_data=None,
            shuffle=True,
            class_weight=None,
            sample_weight=None,
            initial_epoch=0,
            steps_per_epoch=None,
            validation_steps=None,
            validation_batch_size=None,
            validation_freq=1,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False):
        
        """
        Custom fit function for the context model
        
        auto_switch:        Enable/disable autonomous context switching
        train_after_switch: 
        retry_fit:          Locate the next fitting context by re-performing fit.
        absorb:             Reset the switch sequence counter upon successful training.
                            This is mainly used to maintain switch sequencing for temporally-extended tasks
        """

        training._keras_api_gauge.get_cell('fit').set(True)
        # Legacy graph support is contained in `training_v1.Model`.
        version_utils.disallow_legacy_graph('Model', 'fit')
        self._assert_compile_was_called()
        self._check_call_args('fit')

        if validation_split:
            # Create the validation data using the training data. Only supported for
            # `Tensor` and `NumPy` input.
            (x, y, sample_weight), validation_data = (
            data_adapter.train_validation_split((x, y, sample_weight),
                                                validation_split=validation_split,
                                                shuffle=False))

        with self.distribute_strategy.scope(), training_utils.RespectCompiledTrainableState(self):
            # Creates a `tf.data.Dataset` and handles batch and epoch iteration.
            data_handler = WindowedDataHandler(
                x=x,
                y=y,
                sample_weight=sample_weight,
                batch_size=batch_size,
                steps_per_epoch=steps_per_epoch,
                initial_epoch=initial_epoch,
                epochs=epochs,
                shuffle=shuffle,
                class_weight=class_weight,
                max_queue_size=max_queue_size,
                workers=workers,
                use_multiprocessing=use_multiprocessing,
                model=self)

            # Container that configures and calls `tf.keras.Callback`s.
            if not isinstance(callbacks, callbacks_module.CallbackList):
                callbacks = callbacks_module.CallbackList(
                    callbacks,
                    add_history=True,
                    add_progbar=bool(verbose & Verbosity.Progress),
                    model=self,
                    verbose=verbose,
                    epochs=epochs,
                    steps=data_handler.inferred_steps)

            self.stop_training = False
            train_function = self.make_train_function()
            callbacks.on_train_begin()
            self.initialize_fit()
            # Handle fault-tolerance for multi-worker.
            # TODO(omalleyt): Fix the ordering issues that mean this has to
            # happen after `callbacks.on_train_begin`.
            data_handler._initial_epoch = (self._maybe_load_initial_epoch_from_ckpt(initial_epoch))
            for epoch, window_iterator in data_handler.enumerate_epochs():
                self.reset_metrics()
                callbacks.on_epoch_begin(epoch)
                dataset = tf.data.Dataset.zip(next(window_iterator))
                switched_during_epoch = False # Indicate if the model has attempted at least one switch during this epoch
                switched = True               # Indicate if the model switched on the most recent fit iteration
                weights = backend.batch_get_value(self.trainable_variables)
                # Perform a 'fit call'. Assuming retry_fit, this call is re-attempted after each switch until a context fits
                while switched and (retry_fit or not switched_during_epoch):
                    self.initialize_epoch(epoch)
                    iterator = iter(dataset)
                    
                    # Perform a fit call
                    with data_handler.catch_stop_iteration():
                        for step in data_handler.steps():
                            with traceme.TraceMe( 'TraceContext', graph_type='train', epoch_num=epoch, step_num=step, batch_size=batch_size):
                                callbacks.on_train_batch_begin(step)
                                tmp_logs = train_function(iterator)
                                # Catch OutOfRangeError for Datasets of unknown size.
                                # This blocks until the batch has finished executing.
                                # TODO(b/150292341): Allow multiple async steps here.
                                if not data_handler.inferred_steps:
                                    context.async_wait()
                                logs = tmp_logs  # No error, now safe to assign to logs.
                                callbacks.on_train_batch_end(step, logs)
                        
                        switched = not self.update_and_switch(epoch, auto_switch, absorb, retry_fit, verbose)
                        switched_during_epoch |= switched
                        
                        # If a switch occurred, we need to restore the weights
                        if switched or (switched_during_epoch and not train_after_switch):
                            backend.batch_set_value(zip(self.trainable_variables, weights))
                            self.reset_metrics()
                    
                epoch_logs = copy.copy(logs)

                # Run validation.
                if validation_data and self._should_eval(epoch, validation_freq):
                    val_x, val_y, val_sample_weight = (
                        data_adapter.unpack_x_y_sample_weight(validation_data))
                    val_logs = self.evaluate(
                        x=val_x,
                        y=val_y,
                        sample_weight=val_sample_weight,
                        batch_size=validation_batch_size or batch_size,
                        steps=validation_steps,
                        callbacks=callbacks,
                        max_queue_size=max_queue_size,
                        workers=workers,
                        use_multiprocessing=use_multiprocessing,
                        return_dict=True)
                    val_logs = {'val_' + name: val for name, val in val_logs.items()}
                    epoch_logs.update(val_logs)

                callbacks.on_epoch_end(epoch, epoch_logs)
                if self.stop_training:
                    break

            callbacks.on_train_end()
            return self.history
        
        
#     @enable_multi_worker
#     def evaluate(self,
#                    x=None,
#                    y=None,
#                    batch_size=None,
#                    verbose=1,
#                    sample_weight=None,
#                    steps=None,
#                    callbacks=None,
#                    max_queue_size=10,
#                    workers=1,
#                    use_multiprocessing=False,
#                    return_dict=False):
    
#         _keras_api_gauge.get_cell('evaluate').set(True)
#         version_utils.disallow_legacy_graph('Model', 'evaluate')
#         self._assert_compile_was_called()
#         self._check_call_args('evaluate')

#         with self.distribute_strategy.scope():
#             # Creates a `tf.data.Dataset` and handles batch and epoch iteration.
#             data_handler = WindowedDataHandler(
#                 x=x,
#                 y=y,
#                 sample_weight=sample_weight,
#                 batch_size=batch_size,
#                 steps_per_epoch=steps,
#                 initial_epoch=0,
#                 epochs=1,
#                 max_queue_size=max_queue_size,
#                 workers=workers,
#                 use_multiprocessing=use_multiprocessing,
#                 model=self)

#             # Container that configures and calls `tf.keras.Callback`s.
#             if not isinstance(callbacks, callbacks_module.CallbackList):
#                 callbacks = callbacks_module.CallbackList(
#                     callbacks,
#                     add_history=True,
#                     add_progbar=verbose != 0,
#                     model=self,
#                     verbose=verbose,
#                     epochs=1,
#                     steps=data_handler.inferred_steps)

#             test_function = self.make_test_function()
#             callbacks.on_test_begin()
#             for _, iterator in data_handler.enumerate_epochs():  # Single epoch.
#                 self.reset_metrics()
#                 with data_handler.catch_stop_iteration():
#                     for step in data_handler.steps():
#                         with traceme.TraceMe(
#                               'TraceContext',
#                               graph_type='test',
#                               step_num=step):
#                             callbacks.on_test_batch_begin(step)
#                             tmp_logs = test_function(iterator)
#                             # Catch OutOfRangeError for Datasets of unknown size.
#                             # This blocks until the batch has finished executing.
#                             # TODO(b/150292341): Allow multiple async steps here.
#                             if not data_handler.inferred_steps:
#                                 context.async_wait()
#                             logs = tmp_logs  # No error, now safe to assign to logs.
#                             callbacks.on_test_batch_end(step, logs)
#             callbacks.on_test_end()

#             logs = tf_utils.to_numpy_or_python_type(logs)
#             if return_dict:
#                 return logs
#             else:
#                 results = [logs.get(name, None) for name in self.metrics_names]
#                 if len(results) == 1:
#                     return results[0]
#                 return results
        
    def add_context_loss(self, gradients):
        """Calculate and add context loss to context layers"""
        pass
    
    
    def initialize_fit(self):
        """Before training starts..."""
        pass
        
        
    def initialize_epoch(self, epoch):
        """At the beginning of an epoch..."""
        pass
        
        
    def update_and_switch(self, epoch, auto_switch=True, absorb=True, retry_fit=True, verbose=0):
        """
        Update the context layers
        
        Args:
            auto_switch [bool]: Enable/disable autonomous context switching mechanisms
        Return:
            [bool]: Indicate if no switches occurred
        """
        pass
    

class ContextModel(ContextModelBase):
    def __init__(self, *args, **kwargs):
        super(ContextModel, self).__init__(*args, **kwargs)
        self.ctx_layers = [i for i, layer in enumerate(self.layers) if isinstance(layer, Context)]
        
        # We need to map the context layers to their gradient indices
        self.ctx_gradient_map = {}
        index = 0
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Context):
                self.ctx_gradient_map[i] = index # The bias gradient
            index += len(layer.trainable_variables)
    
    
    def _calc_context_loss(self, ctx_layer_idx, gradients):
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
        if self.layers[ctx_layer_idx + 1].use_bias:
            index = self.ctx_gradient_map[ctx_layer_idx] + 1 # Simplify the calculation of the deltas
            delta_at_next_layer = gradients[index]
        else:
            index = self.ctx_gradient_map[ctx_layer_idx]
#             signs = tf.sign(tf.reduce_sum(gradients[index], axis=0))
#             delta_at_next_layer = tf.reduce_sum(tf.multiply(gradients[index], gradients[index]), axis=0)
#             delta_at_next_layer = tf.multiply(signs, tf.reduce_mean(tf.abs(gradients[index]), axis=0))
            delta_at_next_layer = tf.reduce_mean(gradients[index], axis=0)
        transpose_of_weights_at_next_layer = tf.transpose(self.layers[ctx_layer_idx + 1].weights[0])
        context_delta = tf.tensordot(delta_at_next_layer, transpose_of_weights_at_next_layer, 1)
        return context_delta
    
    
    def initialize_fit(self):
        for i in self.ctx_layers:
            if self.layers[i].switch_model:
                self.layers[i].switch_model.on_begin_train()
    
    
    def initialize_epoch(self, epoch):
        # Clear context loss
        for i in self.ctx_layers:
            self.layers[i].clear_context_loss()
        pass
            
    
    def add_context_loss(self, gradients):
        for i in self.ctx_layers:
            self.layers[i].add_context_loss(self._calc_context_loss(i, gradients))
    
    
    def update_and_switch(self, epoch, auto_switch, absorb, retry_fit, verbose):
        updated = True
        for i in reversed(self.ctx_layers):
            layer = self.layers[i]
            updated &= layer.update_and_switch(epoch, auto_switch=auto_switch, absorb=absorb, retry_fit=retry_fit, verbose=verbose)
        return updated
    
    
    def get_contexts(self):
        return [self.layers[layer].hot_context for layer in self.ctx_layers]
    

    def set_contexts(self, contexts):
        for i, layer in enumerate(self.ctx_layers):
            self.layers[layer].hot_context = contexts[i]
            
    # Utility Methods -----------------------------------------------------------------------------
            
    def backup(self):
        """
        A utility function for temporarily backing up a model.
        This function is intended for debugging and hyperparameter tuning purposes.
        """
        self._backup = tf.python.keras.backend.batch_get_value(self.trainable_weights)
        for layer in self.ctx_layers:
            self.layers[layer].backup()
        
        
    def restore(self):
        """
        Restore the temporary backup
        """
        tf.python.keras.backend.batch_set_value(zip(self.trainable_weights, self._backup))
        for layer in self.ctx_layers:
            self.layers[layer].restore()