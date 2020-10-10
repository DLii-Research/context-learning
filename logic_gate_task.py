import datetime
import numpy as np
import pickle
import os
import scipy.stats
import sys
import multiprocessing
import psutil

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Input, Dense

# Import n-task
from keras_context.switches import TdErrorSwitch
from keras_context.callbacks import ContextLogger
from keras_context.flags import Verbosity
from keras_context.layers import Context
from keras_context.models import ContextModel
from keras_context.training import train, evaluate
from keras_context.utils import set_seed

tf.get_logger().setLevel("ERROR")

num_seeds   = int(sys.argv[1]) if len(sys.argv) >= 2 else 1
num_threads = int(sys.argv[2]) if len(sys.argv) >= 3 else 1
num_cycles  = int(sys.argv[3]) if len(sys.argv) >= 4 else 4
num_epochs  = int(sys.argv[4]) if len(sys.argv) >= 5 else 50

gates = {
    "xor":      [0, 1, 1, 0],
    "xnor":     [1, 0, 0, 1],
    "and":      [0, 0, 0, 1],
    "or":       [0, 1, 1, 1],
    "nor":      [1, 0, 0, 0],
    "nand":     [1, 1, 1, 0],
    "custom_1": [1, 0, 1, 0],
    "custom_2": [0, 1, 0, 1]
}

# Build the labels used for `y_train` to work with the `train` convenience function
y_labels = [np.array([[i] for i in gate]) for gate in gates.values()]

# The inputs for each of the gates
x_train = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])

def create_model(optimizer, hrr_size, num_contexts, max_contexts=0, switch_threshold=0.0, add_threshold=0.0, summary=False):
    # Model architecture
    inp = Input((2,))
    x = Dense(hrr_size, activation="relu")(inp)
    x = Context(
        num_contexts,
        TdErrorSwitch(learn_rate=0.5, max_contexts=max_contexts,
                      switch_threshold=switch_threshold, add_threshold=add_threshold),
        name="Gate_Context"
    )(x)
    x = Dense(1, activation="sigmoid")(x)
    model = ContextModel(inputs=inp, outputs=x)

    # Compile the model together with binary_crossentropy loss
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=optimizer,
        metrics=[
            tf.keras.metrics.BinaryAccuracy()
        ]
    )

    # Display the model summary
    if summary:
        model.summary()

    return model

def thread(pid, y_train, seeds, lock, loggers):

    print(pid, psutil.Process().cpu_num())

    for seed in seeds:

        # Hyperparameters
        hrr_size = 256
        switch_threshold = -0.02

        batch_size = 1

        shuffle = True  # Shuffle the active context dataset during training
        task_shuffle = True  # Shuffle the contexts during training after first epoch
        initial_task_shuffle = True  # Shuffle the contexts before training

        verbose = 0  # Verbosity.Contexts

        for num_contexts in (1, 3, 6, 8):

            lock.acquire()

            # Set the seed
            set_seed(seed)

            # Optimizer
            optimizer = tf.keras.optimizers.SGD(1e-1)

            # Create the model
            model = create_model(optimizer, hrr_size, num_contexts=num_contexts,
                            switch_threshold=switch_threshold)

            logger = ContextLogger()
            callbacks = [logger]

            lock.release()

            print(pid, "Seed:", seed, "#ATRs:", num_contexts)

            # Benchmark the model
            history, evals, task_map, context_map = train(model, x_train, y_train, num_cycles, num_epochs, task_shuffle,
                                                        initial_task_shuffle, y_test_list=y_train,
                                                        batch_size=batch_size, shuffle=shuffle, callbacks=callbacks,
                                                        eval_after_cycle=True, verbose=verbose)

            loggers[(seed, num_contexts)] = [logger.to_dict(), evals]

manager = multiprocessing.Manager()

y_train = y_labels[:]
loggers = manager.dict()
lock = multiprocessing.Lock()

threads = []
for i in range(min(num_threads, num_seeds)):
    threads.append(multiprocessing.Process(target=thread, args=(i, y_train, range(i, num_seeds, num_threads), lock, loggers)))
    threads[-1].start()

for th in threads:
    th.join()

with open("./data/logic_gates.dat", 'wb') as f:
    pickle.dump(dict(loggers), f)
