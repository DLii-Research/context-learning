import numpy as np
import tensorflow as tf

def hrr(length):
    """Create a new HRR vector using Tensorflow tensors"""
    length = int(length)      
    shp = int((length-1)/2) 
    x = tf.random.uniform((shp,), -np.pi, np.pi, dtype=tf.dtypes.float32)
    x = tf.cast(x, tf.complex64)
    if length % 2:
        x = tf.concat([tf.ones(1, dtype="complex64"), tf.exp(1j*x), tf.exp(-1j*x[::-1])], axis=0)
    else:  
        x = tf.concat([tf.ones(1, dtype="complex64"),tf.exp(1j*x),tf.ones(1, dtype="complex64"),tf.exp(-1j*x[::-1])],axis=0)
    return tf.math.real(tf.signal.ifft(x))

def hrrs(length, n=1):
    """Create n new HRR vectors using Tensorflow tensors"""
    return tf.stack([hrr(length) for x in range(n)], axis=0)

def circular_conv(x, y):
    """Calculate the circular convolution between two HRR vectors"""
    x = tf.cast(x, tf.complex64)
    y = tf.cast(y, tf.complex64)
    return tf.math.real(tf.signal.ifft(tf.signal.fft(x)*tf.signal.fft(y)))

def circular_conv_power(x, y, p):
    """Calculate the circular convolution between a vector and an HRR raised to the specified power"""
    x = tf.cast(x, tf.complex64)
    y = tf.cast(y, tf.complex64)
    p = tf.cast(p, tf.complex64)
    return tf.math.real(tf.signal.ifft(tf.signal.fft(x)*tf.signal.fft(y)**p))