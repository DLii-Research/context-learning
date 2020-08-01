from tensorflow.keras.callbacks import BaseLogger
import tensorflow as tf

class AtrLogger(tf.keras.callbacks.BaseLogger):
    """
    Log ATR values over time
    """
    
    def __init__(self, logdir, *args, **kwargs):
        super(AtrLogger, self).__init__(*args, **kwargs)
        self.logdir = logdir
        self.writers = {}
        
    def set_model(self, model):
        super(AtrLogger, self).set_model(model)
        self.writers = {self.model.layers[i]: [] for i in self.model.ctx_layers}
        
    def on_epoch_end(self, epoch, logs=None):
        """Create the correct number of writers for the task if necessary"""
        for layer, writers in self.writers.items():
            for i in range(len(writers), layer.num_tasks):
                writers.append(tf.summary.create_file_writer(os.path.join(self.logdir, f"context_atr_{i}"))) # TODO Fix this name here...
            plot_tag = f"context_atr_trace"                                                                  # TODO Fix this name here too...
            for i, writer in enumerate(writers):
                with writer.as_default():
                    value = layer.atr_model.atr_values[i]
                    if value is not None:
                        tf.summary.scalar(plot_tag, data=value, step=epoch)