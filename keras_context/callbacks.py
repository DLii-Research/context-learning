from collections import defaultdict
from tensorflow.keras.callbacks import BaseLogger
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import os

from .models import ContextModel

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