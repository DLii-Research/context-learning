import multiprocessing as mp
import os
import pickle
import queue
import sys
import time
import threading

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Flatten
import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np

from keras_context.switches import SwitchModel, TdErrorSwitch
from keras_context.callbacks import ContextLogger
from keras_context.layers import Context
from keras_context.models import ContextModel
from keras_context.flags import Verbosity
from keras_context.utils import set_seed

tf.get_logger().setLevel("ERROR")

num_seeds    = int(sys.argv[1]) if len(sys.argv) >= 2 else 1
num_threads  = int(sys.argv[2]) if len(sys.argv) >= 3 else 1
num_episodes = int(sys.argv[3]) if len(sys.argv) >= 4 else 1800
switch_freq  = int(sys.argv[4]) if len(sys.argv) >= 5 else 300


def q_function(model, maze_size):
    action_values = np.array([model.predict(np.hstack(
        (np.identity(maze_size), [a]*maze_size))).flatten() for a in np.identity(2)])
    return action_values


def log_q_values(model, maze_size, num_contexts):
    contexts = model.get_contexts()
    values = []
    for i in range(num_contexts):
        model.set_contexts([i])
        values.append(q_function(model, maze_size))
    model.set_contexts(contexts)
    return values


def skip_randint(max_value, skip):
    """
    Generate a new random integer between 0 and `max_value`.
    The returned value will not equal `skip`
    """
    value = np.random.randint(max_value - 1)
    return value if value < skip else (value + 1)


def encode_state_action(state, action):
    return np.concatenate((state, np.identity(2)[action]))


def predict(model, state):
    """
    Predict the Q-values for a given state.
    The resulting predictions are ordered by action ID
    """
    inputs = np.array([np.concatenate((state, a)) for a in np.identity(2)])
    return inputs, model.predict(inputs).flatten()


def policy(predictions, epsilon):
    """
    Epsilon-greedy policy selection
    """
    if np.random.random() < epsilon:
        return np.random.randint(len(predictions)), True
    else:
        return np.argmax(predictions), False


def reward(state, goal):
    if state == goal:
        return 1.0
    return -1.0


def episode(step, model, maze_size, goals, goal_idx, alpha, gamma, epsilon, move_limit, auto_switch=True, ctx_map=None, deltas={}, **kwargs):
    # State encodings
    states = np.identity(maze_size)

    # Pick a random starting state (except goal position)
    state = skip_randint(maze_size, goals[0])

    # Initialize the predicted values for additional program efficiency
    inputs, q = predict(model, states[state])

    # Loop until the goal or move limit is reached
    moves = 0
    while state != goals[goal_idx] and moves < move_limit:

        # Pick an action based on the policy
        action, is_random = policy(q, epsilon)

        # Move to the new state
        prev_state, state = state, (state +
                                    (-1 if action == 0 else 1)) % maze_size

        # Observe the reward
        r = reward(state, goals[goal_idx])
        absorb = state == goals[goal_idx]

        # Predict the action values. This is used to cache results for additional efficiency
        prev_inputs, q_old, (inputs, q) = inputs, q, predict(
            model, states[state])

        # Calculate the target
        discount = 0 if absorb else gamma
        target = q_old[action] + alpha*(r + discount*np.max(q) - q_old[action])

        # Update the model
        target_x = np.array([prev_inputs[action]])
        target_y = np.array([target])

#         print(q_old, target)
        ctx_goal = ctx_map[model.get_contexts()[0]] if not auto_switch else None
        if not auto_switch and ctx_goal != goal_idx and state in (goals[goal_idx], goals[ctx_goal]):
            # Emulated switching for analysis purposes
            model.fit(target_x, target_y, train_after_switch=False, retry_fit=False,
                      absorb=absorb, auto_switch=auto_switch, revert_after_fit=True, **kwargs)
            delta = model.layers[model.ctx_layers[0]].switch_model.delta.numpy()
            model.layers[model.ctx_layers[0]].next_context()
            if state == goals[goal_idx]:
                deltas[step] = (0, delta)
            #     print(
            #         ep, f"Found new goal early at position {state}... Switching to next context {model.get_contexts()[0]}")
            else:
                deltas[step] = (1, delta)
            #     print(
            #         ep, f"Got to an old goal position {state}... Switching to next context {model.get_contexts()[0]}")
        else:
            model.fit(target_x, target_y, train_after_switch=False,
                      retry_fit=False, absorb=absorb, auto_switch=auto_switch, **kwargs)

        moves += 1
        step += 1

#         print()
    return state == goals[goal_idx], moves


# def train(model, maze_size, goals, episodes, alpha, gamma, epsilon, move_limit, switch_freq=100, auto_switch=True, deltas={}, q_values=None, ep_map=[], goal_trace=None, initial_shuffle=False, shuffle=True, true_random=False, **kwargs):
#     step = 0
#     if initial_shuffle:
#         if true_random:
#             goal_idx = np.random.randint(len(goals))
#     else:
#     goal_idx = 0
#     goal_map = np.arange(len(goals))
#     if initial_shuffle:
#         np.random.shuffle(goal_map)
#         goal_map = list(goal_map)
#         goal_idx = goal_map.pop()

#     # Create the context map
#     ctx_map = reversed(goal_map.copy())
#     if goal_trace is not None:
#         goal_trace.append(goal_map[goal_idx])

#     for ep in range(episodes):
#         # Execute an episode
#         ep_map.append(step)
#         completed, moves = episode(step, model, maze_size, goals,
#                                    goal_idx, alpha, gamma, epsilon, move_limit, auto_switch=auto_switch, ctx_map=ctx_map, deltas=deltas, **kwargs)
#         step += moves

#         # Log q-values
#         if q_values is not None:
#             q_values[ep] = q_function(model, maze_size)

#         # Change goals every so often
#         if ep % switch_freq == 0 and ep > 0:
#             if goal_idx >= len(goals) - 1:
#                 goal_idx = 0
#                 if shuffle:
#                     last_idx = goal_map[-1]
#                     np.random.shuffle(goal_map)
#                     while goal_map[0] == last_idx:
#                         np.random.shuffle(goal_map)
#             else:
#                 goal_idx += 1
#             if goal_trace is not None:
#                 goal_trace.append(goal_map[goal_idx])
#     return ctx_map


def train(model, maze_size, goals, episodes, alpha, gamma, epsilon, move_limit, switch_freq=100, auto_switch=True, deltas={}, q_values=None, ep_map=[], goal_trace=None, initial_shuffle=False, shuffle=True, true_random=False, monitor=None, **kwargs):
    step = 0
    ctx_map = None
    if true_random:
        goal_idx = np.random.randint(len(goals))
        goal_indices = []
    else:
        goal_indices = list(np.arange(len(goals)))
        if initial_shuffle:
            np.random.shuffle(goal_indices)
        ctx_map = list(reversed(goal_indices.copy()))
        goal_idx = goal_indices.pop()
        
    if goal_trace is not None:
        goal_trace.append(goal_idx)
    
    for ep in range(episodes):
        # Execute an episode
        if monitor is not None:
            monitor(ep)
            
        ep_map.append(step)
        completed, moves = episode(step, model, maze_size, goals,
                                   goal_idx, alpha, gamma, epsilon, move_limit, auto_switch=auto_switch, ctx_map=ctx_map, deltas=deltas, **kwargs)
        step += moves

        # Log q-values
        if q_values is not None:
            q_values[ep] = log_q_values(model, maze_size, len(goals))

        # Change goals every so often
        if ep % switch_freq == 0 and ep > 0:
            if shuffle:
                if len(goal_indices) > 0:
                    goal_idx = goal_indices.pop()
                else:
                    last_idx = goal_idx
                    while last_idx == goal_idx:
                        goal_idx = np.random.randint(len(goals))
            else:
                goal_idx = (goal_idx + 1) % len(goals)
            if goal_trace is not None:
                goal_trace.append(goal_idx)
    
    if q_values is not None:
        q_values[ep] = log_q_values(model, maze_size, len(goals))
    return ctx_map


def create_model(num_atrs, maze_size, hrr_size, use_bias):

    # Optimizer
    optimizer = tf.keras.optimizers.SGD(1e-2)

    # switch_threshold = -0.04
    learn_rate = 0.003
    switch_threshold = -0.06
    add_threshold = -0.06

    # Model Architecture
    inp = Input((maze_size + 2,))
    x = Dense(hrr_size, activation="relu", use_bias=use_bias)(inp)
    x = Context(num_atrs, TdErrorSwitch(learn_rate, switch_threshold, switch_delay=250, initial_loss=5.0))(x)
    x = Dense(1, activation="linear", use_bias=use_bias)(x)

    # Create the model
    model = ContextModel(inputs=inp, outputs=x)

    # Compile the model
    model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=optimizer)

    return model


def thread(pid: int, q: mp.Queue, n: int, loggers: dict, progress: mp.Value, lock: mp.Lock):

    # Maze Settings
    maze_size = 10
    goals = (0, 1, 5)

    epsilon = 0.3
    alpha = 1.0
    gamma = 1.0
    move_limit = 30

    while not q.empty():
        try:
            group = q.get()
            seed, num_atrs = group

            set_seed(seed)
            
            hrr_size = 1024
            use_bias = True
            model = create_model(num_atrs, maze_size, hrr_size, use_bias)
            logger = ContextLogger(track_epochs_internally=True)
            callbacks = [logger]

            deltas = {}
            q_values = {}
            goal_map = []
            ep_map = []
            monitor = lambda x: setattr(progress, 'value', x)
            train(model, maze_size, goals, num_episodes, alpha, gamma, epsilon, move_limit, switch_freq, auto_switch=True, deltas=deltas,
                                  q_values=q_values, ep_map=ep_map, goal_trace=goal_map, initial_shuffle=True, shuffle=True, monitor=monitor,
                                  true_random=False, callbacks=callbacks, verbose=0)

            lock.acquire()
            progress.value += 1
            loggers[group] = [logger.to_dict(), deltas, q_values, goal_map, ep_map]
            lock.release()
        except queue.Empty:
            return    
        
finished = False

def monitor_thread(progress, q):
    while not finished:
        output = f"Jobs in queue: {q.qsize()}\n"
        for thread, prog in enumerate(progress):
            output += f"Thread {thread}: {prog.value}/{num_episodes}\n"
#         output += f"\nEstimated time remaining: "
        print("\033[H\033[J" + output) # Clear the screen and output
        time.sleep(1)

        
# Build the queue
q = mp.Queue()
for seed in range(20, 20+num_seeds):
# for seed in (7,):
    for num_atrs in (3,):
        q.put_nowait((seed, num_atrs))
n = q.qsize()

print("Using", num_threads, "threads.")
print(f"Runs completed: {n-q.qsize()}/{n}", end="\r")

manager = mp.Manager()
loggers = manager.dict()
progresses = []
lock = mp.Lock()
threads = []
start = time.time()

for i in range(num_threads):
    progress = manager.Value('i', 0)
    progresses.append(progress)
    threads.append(mp.Process(target=thread, args=[i, q, n, loggers, progress, lock]))
    threads[-1].start()

mon_thread = threading.Thread(target=monitor_thread, args=[progresses, q])
mon_thread.start()
    
for th in threads:
    th.join()

finished = True
# mon_thread.join()
    
print("Finished")

with open("./data/maze_analysis9.dat", 'wb') as f:
    pickle.dump(dict(loggers), f)
