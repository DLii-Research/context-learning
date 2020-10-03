import numpy as np
import tensorflow as tf

import matplotlib
import random
import struct

def hrr(length, normalized=True):
    """Create a new HRR vector using Tensorflow tensors"""
    length = int(length)      
    shp = int((length-1)/2)
    if normalized:    
        x = tf.random.uniform( shape = (shp,), minval = -np.pi, maxval = np.pi, dtype = tf.dtypes.float32, seed = 100, name = None )
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
    
    
def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    
def trace(label, value, style='-', color=None, condition=True):
    if not condition:
        return None
    return (label, value, style, color)
    
    
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
    
def idx_load(filename):
    IDX_DATA_TYPE = {
        0x08: 'B',
        0x09: 'b',
        0x0B: 'h',
        0x0C: 'i',
        0x0D: 'f',
        0x0E: 'd',
    }
    DATA_SIZE = {
        'B': 1,
        'b': 1,
        'h': 2,
        'i': 4,
        'f': 4,
        'd': 8,
    }
    decode = lambda x: int.from_bytes(x, byteorder="big")
    with open(filename, 'rb') as f:
        # Discard the first two bytes
        f.read(2)
        
        # Get the data type
        data_type = IDX_DATA_TYPE[decode(f.read(1))]
        data_size = DATA_SIZE[data_type]
        
        # Grab the number of dimensions
        dim = decode(f.read(1))
        
        # Sizes of each dimension
        sizes = [decode(f.read(4)) for size in range(dim)]
        
        # The result array
        result = np.zeros(sizes)
        
        indices = [0]*len(sizes)
        end = len(indices) - 1
        while indices[0] < sizes[0]:
            [result[tuple(indices)]] = struct.unpack(f">{data_type}", f.read(data_size))
            indices[end] += 1
            if indices[end] == sizes[end]:
                i = end
                while i > 0 and indices[i] >= sizes[i]:
                    indices[i] = 0
                    i -= 1
                    indices[i] += 1
        if data_type != 'f':
            return result.astype(int)
        return result