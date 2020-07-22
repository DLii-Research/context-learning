from IPython.display import clear_output
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf
import time


def copy_metric(metric):
    """Copy a new metric instance from an existing instance"""
    return tf.keras.metrics.deserialize(tf.keras.metrics.serialize(metric))


# Display a progress bar. (clears all output)
# @param {Integer} progress
# @param {String}  title
def display_progress(progress, title = None):
    WIDTH = 20
    progress = max(0, min(1.0, progress))
    blocks = math.floor(20 * progress)
    clear_output(wait = True)
    if title:
        print(title)
    print(f"Progress: [{'#'*blocks}{'-'*(WIDTH - blocks)}] {progress*100:.2f}%")


def hrr(length, normalized=True):
    """Create a new HRR vector using Tensorflow tensors"""
    length = int(length)      
    shp = int((length-1)/2)
    if normalized:    
        x = tf.random.uniform( shape = (shp,), minval = -math.pi, maxval = math.pi, dtype = tf.dtypes.float32, seed = 100, name = None )
        x = tf.cast(x, tf.complex64)
        if length % 2:
            x = tf.math.real( tf.signal.ifft( tf.concat([tf.ones(1, dtype="complex64"), tf.exp(1j*x), tf.exp(-1j*x[::-1])], axis=0)))

        else:  
            x = tf.math.real(tf.signal.ifft(tf.concat([tf.ones(1, dtype="complex64"),tf.exp(1j*x),tf.ones(1, dtype="complex64"),tf.exp(-1j*x[::-1])],axis=0)))
    else:        
        x = tf.random.normal( shape = (length,), mean=0.0, stddev=1.0/tf.sqrt(float(length)),dtype=tf.dtypes.float32,seed=100,name=None)
    return x


def hrrs(length, n=1, normalized=True):
    """Create n new HRR vectors using Tensorflow tensors"""
    return tf.stack([hrr(length, normalized) for x in range(n)], axis=0)


def circ_conv(x, y):
    """Calculate the circular convolution between two HRR vectors"""
    x = tf.cast(x, tf.complex64)
    y = tf.cast(y, tf.complex64)
    return tf.math.real(tf.signal.ifft(tf.signal.fft(x)*tf.signal.fft(y)))


def logmod(x):
    return np.sign(x)*np.log(abs(x) + 1)


def plot(title, labels, *frameGroups):
    fig, ax = plt.subplots()
    plotFrames(ax, title, labels, *frameGroups, xlabel="Epoch", ylabel="Value")
    ax.grid()
    plt.legend()
    
    
def plotFrames(ax, title, labels, *frameGroups, xlabel=None, ylabel=None):
    for i, frames in enumerate(frameGroups):
        keys = tuple(frames.keys() if type(frames) == dict else range(len(frames)))
        t = np.arange(keys[0], keys[-1] + 1, 1)
        ax.plot(t, list(frames.values()), label=(labels[i] if labels else None))
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)