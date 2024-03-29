import tensorflow as tf
import tensorflow.keras as keras
from keras.engine import data_adapter

from .layers import ContextLayerBase

class ContextModelBase(keras.Model):
    def __init__(self, *args, retry_fit=True, verbose=0, **kwargs):
        super(ContextModelBase, self).__init__(*args, **kwargs)
        self.retry_fit = retry_fit
        self.verbose = verbose
        self.context_layer_map = [i for i, l in enumerate(self.layers) if isinstance(l, ContextLayerBase)]
        self.context_gradient_map = self._build_gradient_map()
        self.context_layers = [self.layers[i] for i in self.context_layer_map]
        self.num_context_layers = tf.constant(len(self.context_layers))
        self.prev_epoch = tf.Variable(-1, dtype=tf.int32, trainable=False, name="Previous Epoch")
        self.is_retry = tf.Variable(False, dtype=tf.bool, trainable=False, name="Is Retry")
        
        # Store the context deltas at each epoch
        self.context_deltas = tf.Variable(tf.zeros(self.num_context_layers), dtype=tf.float32, trainable=False, name="Context Deltas")
        
        # Set context layer verbosity
        for layer_id in self.context_layer_map:
            layer = self.layers[layer_id]
            layer.verbose = max(layer.verbose, verbose)
        
    def _build_gradient_map(self):
        gradient_map = []
        index = 0
        for i, layer in enumerate(self.layers):
            if isinstance(layer, ContextLayerBase):
                # Offset if we can use the bias weight
                offset = int(self.layers[i + 1].use_bias)                
                gradient_map.append(index + offset)
            index += len(layer.trainable_variables)
        return gradient_map
            
    def _calc_context_loss(self, context_layer_id, gradients):
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
        # If we aren't using the bias, reduce the dimensionality using mean square
        layer_index = self.context_layer_map[context_layer_id]
        if len(gradients.shape) > 1:
            gradients = tf.reduce_mean(tf.multiply(gradients, gradients), axis=0)
        next_layer_weights =  tf.transpose(self.layers[layer_index + 1].weights[0])
        context_delta = tf.tensordot(tf.expand_dims(gradients, 0), next_layer_weights, 1)
        return context_delta
    
    def add_observed_loss(self, gradients):
        for i, grads in enumerate(gradients):
            observed_loss = self._calc_context_loss(i, grads)
            self.layers[self.context_layer_map[i]].add_observed_loss(observed_loss)
            
    def backup(self):
        self._backup = keras.backend.batch_get_value(self.trainable_weights)
        
    def restore(self):
        keras.backend.batch_set_value(zip(self.trainable_weights, self._backup))
            
    def perform_epoch(self, epoch, absorb=True):
        def begin_epoch():
            self.prev_epoch.assign(epoch)
            self.is_retry.assign(False)
            self.backup()
            return True
        def update_and_switch():
            switched = False
            for layer in self.context_layers:
                switched = tf.logical_or(layer.update(absorb, self.is_retry), switched)
            if not self.is_retry:
                self.context_deltas.assign([layer.delta for layer in self.context_layers])
            tf.cond(switched, self.restore, tf.no_op)
            self.is_retry.assign(True)
            return switched
        return tf.cond(self.prev_epoch != epoch, begin_epoch, lambda: tf.logical_and(update_and_switch(), self.retry_fit))
    
    @property
    def non_trainable_weights(self):
        weights = super(ContextModelBase, self).non_trainable_weights
        return [w for w in weights if hasattr(w, "shape") and w.shape.dims != None]
    
class ContextModel(ContextModelBase):
    pass