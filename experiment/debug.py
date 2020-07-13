from tensorflow.python.keras.engine import data_adapter
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import Model
import tensorflow as tf

import numpy as np
import random

class NTaskModel(Model):

    def compile(self, *args, **kwargs):
        super(NTaskModel, self).compile(*args, **kwargs)
        self.context_layers = []
        self.my_test_count = 0


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


    def _forward_pass(self, x, y, sample_weight=None):
        """
        Performs a forward pass with active switching mechanism
        x: input data
        y: expected output (required for switching mechanisms)
        """

        # Perform forward pass and calculate loss
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, sample_weight, regularization_losses=self.losses)

        # Extract the gradients for the loss of the current sample
        gradients = tape.gradient(loss, self.trainable_variables)

        for context_layer_idx in self.context_layers:
            self.layers[context_layer_idx].context_loss += self._calc_context_loss(context_layer_idx, gradients)

        # Return the calculated gradients
        return y_pred, gradients


    def train_step(self, batch):

        self.my_test_count += 1
        tf.print("Train step", self.my_test_count)
        print(self.my_test_count, batch)

        # Unpack the data
        data = data_adapter.expand_1d(batch)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(batch)

        # Perform a forward pass and calculate gradients
        y_pred, gradients = self._forward_pass(x, y, sample_weight)

        # Apply the gradients to the model
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Update the metrics
        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        return {metric.name: metric.result() for metric in self.metrics}


    def fit(self, *args, **kwargs):
        # Method 1: Modify batch size here to size of dataset, divide in train step
        return super(NTaskModel, self).fit(*args, **kwargs)

inp = Input((2,))
x = Dense(40, activation="relu")(inp)
x = Dense(1, activation="sigmoid")(x)
model = NTaskModel(inputs=inp, outputs=x)

model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss=tf.keras.losses.BinaryCrossentropy(), metrics=["accuracy"])

x_train = np.array([[1, 1], [-1, 1], [2, 2], [3, 3]]) # 4 inputs
y_train = np.array([[1], [0], [2], [3]])              # 4 labels

model.fit(x_train, y_train, batch_size=1, shuffle=False, verbose=0)
