from collections import defaultdict
from tensorflow.keras.callbacks import BaseLogger
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import os

from .models import ContextModel

# class AtrLoggerTensorBoard(tf.keras.callbacks.BaseLogger):
#     """
#     Log ATR models via TensorBoard
#     """
    
#     def __init__(self, logdir, *args, **kwargs):
#         super(AtrLoggerTensorBoard, self).__init__(*args, **kwargs)
#         self.logdir = logdir
#         self.writers = {}
        
#     def set_model(self, model):
#         super(AtrLoggerTensorBoard, self).set_model(model)
#         self.writers = {self.model.layers[i]: [] for i in self.model.ctx_layers}
        
#     def on_epoch_end(self, epoch, logs=None):
#         """Create the correct number of writers for the task if necessary"""
#         for layer, writers in self.writers.items():
#             for i in range(len(writers), layer.num_contexts):
#                 writers.append(tf.summary.create_file_writer(os.path.join(self.logdir, f"{layer.name}_Atr_{i}")))
#             plot_tag = f"{layer.name}_Atr_Trace"
#             for i, writer in enumerate(writers):
#                 with writer.as_default():
#                     value = layer.switch_model.values[i]
#                     if value is not None:
#                         tf.summary.scalar(plot_tag, data=value, step=epoch)

                        
class ContextLogger(tf.keras.callbacks.BaseLogger):
    """
    Log context switch models via matplotlib
    """
    
    def __init__(self, track_epochs_internally=False, *args, **kwargs):
        super(ContextLogger, self).__init__(*args, **kwargs)
        self.plots = None # layer_name -> { plot_name -> { trace_name -> { data } } }
        self.model = None
        self.epoch = 0 if track_epochs_internally else None
        
    def plot(self, vertical=True, figsize=(30, 9), title=None, savefile=None):
        if savefile and savefile.endswith(".pgf"):
            matplotlib.use("pgf")
            matplotlib.rcParams.update({
                "pgf.texsystem": "pdflatex",
                "font.family": "serif",
                "text.usetex": True,
                "pgf.rcfonts": False
            })
        for layer, plots in self.plots.items():
            dim = (len(plots), 1) if vertical else (1, len(plots))
            fig, axs = plt.subplots(*dim, figsize=figsize, sharey=False, sharex=True)
            for i, ((plot_name, xlabel, ylabel), traces) in enumerate(plots.items()):
                for label, trace in traces.items():
                    axs[i].plot(
                        trace["x"], trace["y"],
                        label=label,
                        color=trace["color"],
                        linestyle=trace["style"])
                axs[i].set_title(plot_name)
                axs[i].set_xlabel(xlabel)
                axs[i].set_ylabel(ylabel)
                axs[i].grid(True)
                axs[i].legend()
            fig.suptitle(title)
        plt.tight_layout()
        if savefile:
            plt.savefig(savefile)
    
    def set_model(self, model):
        if not isinstance(model, ContextModel):
            return
        if self.model != model:
            self.plots = {}
            for i in model.ctx_layers:
                layer = model.layers[i]
                if layer.switch_model is not None:
                    self.plots[layer] = defaultdict(lambda: defaultdict(lambda: {
                        "x": [],
                        "y": [],
                        "style": None,
                        "color": None
                    }))
        super(ContextLogger, self).set_model(model)
        
    def on_epoch_end(self, epoch, logs=None):
        if self.epoch is not None:
            self.epoch += 1
        for layer, plots in self.plots.items():
            for plot_name, traces in layer.switch_model.epoch_traces(epoch).items():
                for trace_data in traces:
                    if trace_data is not None:
                        (label, value, style, color) = trace_data
                        trace = plots[plot_name][label]
                        if len(trace["x"]) == 0:
                            trace["style"] = style
                            trace["color"] = color
                        trace["x"].append(self.epoch or epoch)
                        trace["y"].append(value)
       
    
    def to_dict(self):
        result = {}
        for layer, plots in self.plots.items():
            result[layer.name] = {}
            for key, data in plots.items():
                result[layer.name][key] = dict(data)
                for label in result[layer.name][key]:
                    result[layer.name][key][label]['y'] = [float(v) for v in result[layer.name][key][label]['y']]
        return result