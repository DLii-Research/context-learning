from PyQt5.QtCore import qAbs
import numpy as np
import os
import multiprocessing as mp
import queue
import pickle
import psutil
import sys
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, Dropout

# Import n-task
from keras_context.switches import TdErrorSwitch
from keras_context.callbacks import ContextLogger
from keras_context.flags import Verbosity
from keras_context.layers import Context
from keras_context.models import ContextModel
from keras_context.training import train, evaluate
from keras_context.utils import set_seed, idx_load

tf.get_logger().setLevel("ERROR")

# Training parameters
num_seeds   = int(sys.argv[1]) if len(sys.argv) >= 2 else 1
num_threads = int(sys.argv[2]) if len(sys.argv) >= 3 else 1
num_train   = int(sys.argv[3]) if len(sys.argv) >= 4 else 5000
num_test    = int(sys.argv[4]) if len(sys.argv) >= 5 else 1000
num_cycles  = int(sys.argv[5]) if len(sys.argv) >= 6 else 5
num_epochs  = int(sys.argv[6]) if len(sys.argv) >= 7 else 5

# Training images and labels
training_images = idx_load("./datasets/mnist/train-images.idx3-ubyte")[:num_train]
training_labels = idx_load("./datasets/mnist/train-labels.idx1-ubyte")[:num_train]

# Testing images and labels
testing_images = idx_load("./datasets/mnist/t10k-images.idx3-ubyte")[:num_test]
testing_labels = idx_load("./datasets/mnist/t10k-labels.idx1-ubyte")[:num_test]

# Normalize inputs
training_images = training_images / np.max(training_images)
testing_images = testing_images / np.max(testing_images)

def create_divisibility_labels(labels):
    # Even/Odd/divisible by 3
    return (labels % 2 == 0).astype(int), (labels % 2 == 1).astype(int), (labels % 3 == 0).astype(int)

# Training Data
x_train = training_images.reshape(training_images.shape + (1,))
y_train_list = create_divisibility_labels(training_labels)

# Testing Data
x_test = testing_images.reshape(testing_images.shape + (1,))
y_test_list = create_divisibility_labels(testing_labels)

def create_model(num_contexts):

	# Optimizer
	optimizer = tf.keras.optimizers.Adam(1e-3)

	# Hyperparameters
	hrr_size = 128
	switch_threshold = -0.005

	# Model Architecture
	inp = Input(x_train.shape[1:])
	x = Conv2D(64, kernel_size=(8, 8), activation="relu")(inp)
	x = Conv2D(128, (8, 8), activation="relu")(x)
	x = MaxPooling2D(pool_size=(2, 2))(x)
	x = Dropout(0.25)(x)
	x = Flatten()(x)
	x = Dense(hrr_size, activation="relu")(x)
	x = Dropout(0.5)(x)
	x = Context(num_contexts, TdErrorSwitch(0.5, switch_threshold))(x)
	x = Dense(1, activation="sigmoid")(x)

	# Create the model
	model = ContextModel(inputs=inp, outputs=x)

	# Compile the model
	model.compile(loss="bce", optimizer=optimizer, metrics=["accuracy"])

	return model


def thread(pid, q, loggers, lock):
	batch_size = 32

	shuffle = True # Shuffle the active context dataset during training
	task_shuffle = True # Shuffle the contexts during training after first epoch
	initial_task_shuffle = True # Shuffle the contexts before training

	verbose = 0
	while not q.empty():
		try:
			seed, num_atrs = q.get()

			set_seed(seed)
			model = create_model(num_atrs)

			logger = ContextLogger()
			callbacks = [logger]

			# Train the model
			print("Training; Seed:", seed, "#ATRs", num_atrs)
			history, evals, task_map, context_map = train(model, x_train, y_train_list, num_cycles, num_epochs, task_shuffle, initial_task_shuffle, x_test=x_test,
												y_test_list=y_test_list, eval_after_cycle=True, batch_size=batch_size, shuffle=shuffle, callbacks=callbacks, verbose=verbose)

			lock.acquire()
			loggers[(seed, num_atrs)] = [logger.to_dict(), evals]
			lock.release()

		except queue.Error:
			print("Queue error")
			return
	print(pid, "Finished")

# Build the queue
q = mp.Queue()
for seed in range(num_seeds):
	for num_atrs in (1, 2, 3):
		q.put_nowait((seed, num_atrs))

manager = mp.Manager()
loggers = manager.dict()
lock = mp.Lock()
threads = []
for i in range(num_threads):
	threads.append(mp.Process(target=thread, args=[i, q, loggers, lock]))
	threads[-1].start()

for th in threads:
	th.join()

with open("./data/mnist.dat", 'wb') as f:
    pickle.dump(dict(loggers), f)
