import itertools
import multiprocessing as mp
import numpy as np
import os
import pickle
import queue
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras.layers import Input, Dense, Flatten
import tensorflow as tf

from keras_context.callbacks import ContextLogger
from keras_context.flags import Verbosity
from keras_context.layers import Context
from keras_context.models import ContextModel
from keras_context.switches import TdErrorSwitch
from keras_context.utils import set_seed

tf.get_logger().setLevel("ERROR")

num_seeds    = int(sys.argv[1]) if len(sys.argv) >= 2 else 1
num_threads  = int(sys.argv[2]) if len(sys.argv) >= 3 else 1
num_episodes = int(sys.argv[3]) if len(sys.argv) >= 4 else 100*12

def random_card():
    """
    Generate a one-hot encoded card.
    Returns a tuple of the card and the expected output label
    """
    x = np.identity(3)
    np.random.shuffle(x)
    return x


def policy(predicted, epsilon):
    if np.random.random() < epsilon:
        return np.random.randint(3), True
    return np.argmax(predicted), False


def reward(sort, rule):
    if sort == rule:
        return 1.0
    return -1.0


def episode(model, ctx_map, rule, ep, epsilon=0.01, alpha=1.0, auto_switch=True, deltas={}, **kwargs):

    # Action choices
    actions = np.identity(3)

    # Random card to sort
    card = random_card()

    # Create the possible inputs to feed into the neural network
    inputs = np.array([np.vstack((a, card)) for a in actions])

    # Predict and sort
    values = model.predict(inputs)
    action, is_random = policy(values.flatten(), epsilon)
    r = reward(action, rule)

    # Q-learning update
    q_old = values[action][0]
    target = np.array([[q_old + alpha*(r - q_old)]])

    # Update the model
    x = np.array([np.vstack((actions[action], card))])
    if not auto_switch and ctx_map[model.get_contexts()[0]] != rule and action in (ctx_map[model.get_contexts()[0]], rule):
        old_ctx = model.get_contexts()[0]
        model.fit(x, target, train_after_switch=False, retry_fit=False,
                  revert_after_fit=True, auto_switch=False, **kwargs)
        deltas[ep] = model.layers[model.ctx_layers[0]
                                  ].switch_model.delta.numpy()
        model.layers[model.ctx_layers[0]].next_context()
        # if action == rule:
        #     print(
        #         f"Found the correct context early through an exploratory move. Switch to next sequential context: {model.get_contexts()[0]}")
        # else:
        #     print(
        #         f"Determined context is incorrect... Trying the next context: {model.get_contexts()[0]}")
    else:
        model.fit(x, target, train_after_switch=False,
                  retry_fit=False, auto_switch=auto_switch, **kwargs)

    return action == rule, is_random


def all_possible_inputs():
    result = []
    for card in itertools.permutations(np.identity(3)):
        for action in np.identity(3):
            result.append(np.vstack((action, card)))
    return np.array(result)


def evaluate_model(model):
    # Get the current model state
    contexts = model.get_contexts()

    # Generate all possible input combinations
    inputs = all_possible_inputs()

    accuracy_map = []
    for context in range(3):
        model.set_contexts([context])
        predictions = model.predict(inputs).reshape((6, 3, 1))
        accuracy_map.append(np.bincount(
            np.argmax(predictions, axis=1).flatten(), minlength=3) / 6)

    # Brute force best-fit because I'm too lazy :p
    best_accuracy = 0
    for mapping in itertools.permutations(range(3)):
        accuracy = np.mean([accuracy_map[i][j] for i, j in enumerate(mapping)])
        if accuracy > best_accuracy:
            best_accuracy = accuracy

    # Restore the model state
    model.set_contexts(contexts)

    return best_accuracy


def wisconsin_card_sort(model, episodes, switch_freq=75, epsilon=0.01, alpha=1.0, auto_switch=False, initial_shuffle=False, shuffle=True, deltas={}, **kwargs):
    NUM_DIMENSIONS = 3
    rules = np.arange(NUM_DIMENSIONS)
    if initial_shuffle:
        np.random.shuffle(rules)
    ctx_map = rules.copy()

    rule, rules = rules[0], rules[1:]
    eval_accuracy = []
    for ep in range(episodes):
        # Perform the episode. Count the sequentially correct episodes
        is_correct, is_random = episode(
            model, ctx_map, rule, ep, epsilon, alpha, auto_switch=auto_switch, deltas=deltas, **kwargs)

        eval_accuracy.append(evaluate_model(model))

        # Change the rule once the model has learned the rule
        if ep > 0 and ep % switch_freq == 0:
            if shuffle:
                if len(rules) == 0:
                    old_rule = rule
                    while rule == old_rule:
                        rule = np.random.randint(NUM_DIMENSIONS)
                else:
                    rule, rules = rules[0], rules[1:]
            else:
                rule = (rule + 1) % NUM_DIMENSIONS

    return eval_accuracy


def create_model(num_atrs):
	# Optimizer
	optimizer = tf.keras.optimizers.SGD(1e-1)

	# Hyperparameters
	hrr_size = 512
	switch_threshold = -8e-6

	# Model Architecture
	inp = Input((4, 3))
	x = Flatten()(inp)
	x = Dense(hrr_size, activation="relu", use_bias=False)(x)
	x = Context(num_atrs, TdErrorSwitch(
		0.05, switch_threshold, initial_loss=20e-5))(x)
	x = Dense(1, activation="linear", use_bias=False)(x)

	# Create the model
	model = ContextModel(inputs=inp, outputs=x)

	# Compile the model
	model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=optimizer)

	return model


def thread(pid: int, q: mp.Queue, loggers: dict, lock: mp.Lock):

	switch_frequency = 100
	epsilon = 0.1
	alpha = 1.0

	while not q.empty():
		try:
			seed, num_atrs = q.get()

			set_seed(seed)

			model = create_model(num_atrs)
			logger = ContextLogger(track_epochs_internally=True)
			callbacks = [logger]

			deltas = {}
			accuracy = wisconsin_card_sort(model, num_episodes, switch_frequency, epsilon, alpha, auto_switch=True, deltas=deltas,
			initial_shuffle=True, shuffle=True, callbacks=callbacks, verbose=0)

			lock.acquire()
			loggers[(seed, num_atrs)] = [logger.to_dict(), deltas, accuracy]
			print(f"Runs remaining: {q.qsize()}", end="\r")
			lock.release()
		except queue.Empty:
			return

# Build the queue
q = mp.Queue()
for seed in range(num_seeds):
	for num_atrs in (3,):
		q.put_nowait((seed, num_atrs))

print("Using", num_threads, "threads.")

print(f"Runs remaining: {q.qsize()}", end="\r")

manager = mp.Manager()
loggers = manager.dict()
lock = mp.Lock()
threads = []

for i in range(num_threads):
	threads.append(mp.Process(target=thread, args=[i, q, loggers, lock]))
	threads[-1].start()

for th in threads:
	th.join()

print("Finished")

with open("./data/wcst.dat", 'wb') as f:
    pickle.dump(dict(loggers), f)
