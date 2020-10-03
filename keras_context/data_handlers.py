from tensorflow.python.keras.engine import data_adapter

import numpy as np

# Extended from https://github.com/tensorflow/tensorflow/blob/v2.2.0/tensorflow/python/keras/engine/data_adapter.py
class WindowedDataHandler(data_adapter.DataHandler):
    """
    Enumerating over this data handler yields windows of the dataset.
    This is important for n-task because if a context switch occurs
    during an epoch the data needs to be sent back through the network.
    """
    def calc_window_size(self):
        batch_size = self._adapter.batch_size()
        num_samples = self._inferred_steps*batch_size
        if self._adapter.has_partial_batch():
            num_samples -= batch_size - self._adapter.partial_batch_size()
        return np.ceil(num_samples/min(batch_size, num_samples))
    
    def enumerate_epochs(self):
        data_iterator = iter(self._dataset.window(self.calc_window_size()))
        for epoch in range(self._initial_epoch, self._epochs):
            if self._insufficient_data:
                break
            if self._adapter.should_recreate_iterator():
                data_iterator = iter(self._dataset.window(self.calc_window_size()))
            yield epoch, data_iterator
            self._adapter.on_epoch_end()