import tensorflow as tf
import numpy as np

def train(model,
          x_train,
          y_train_list,
          cycles=1,
          epochs=1,
          task_shuffle=True,
          initial_task_shuffle=False,
          explicit_contexts=None,
          assert_contexts=False,
          y_test_list=None,
          eval_after_cycle=False,
          **kwargs):
    """
    Train an NTask model on a dataset containing samples from multiple contexts.

    Args:
      model            : The NTask model instance
      x_train          : The input data
      y_train_list     : A list of tuples containing the y_train value and context IDs.
                         e.g. (y_train, 0, 2) # (y_train, context_layer_0 = 0, context_layer_1 = 2)
      cycles           : The number of iterations over the entire y_train_list
      epochs           : The number of epochs to perform on each y_train element within y_train_list
      task_shuffle     : Shuffle the y_train_list on each cycle (does not shuffle data within the task)
      initial_shuffle  : Shuffle the y_train_list on the first epoch. If set to false, context order
                         within the context layers is guaranteed to remain in the same order as provided
      explicit_contexts: A list of context mappings for each task. If none (either entirely, or
                         contains None as an item), the current task will be dynamically switched.
      **kwargs         : All other keyword arguments are passed to `model.fit`
    """

    # Validate provided explicit contexts
    if explicit_contexts is None:
        explicit_contexts = [None]*len(y_train_list)
    elif len(explicit_contexts) != len(y_train_list):
        raise ValueError(f"Supplied number of explicit contexts ({len(explicit_contexts)}) does not match the number of tasks ({len(y_train_list)}).")
    else:
        for i, task in enumerate(explicit_contexts):
            if task is not None and len(task) != len(model.context_layers):
                raise ValueError(f"Provided explicit contexts for task {i} does not match the number of context layers.")

    # Create an index map of the tasks
    indices = np.arange(len(y_train_list))

    # Shuffle if necessary
    if initial_task_shuffle:
        np.random.shuffle(indices)

    # Map the shuffled tasks in the order they are passed to the `fit` method
    task_map = indices.copy()

    # Track the layer contexts for each task for later evaluation
    context_map = [None]*len(indices)

    # Keep track of the model's history
    history_list = []
    eval_list = []

    for cycle in range(cycles):
        for i, task in enumerate(indices):
            y_train = y_train_list[task]
            y_test = y_test_list[task] if y_test_list is not None else None

            # Calculate the initial epoch to start training on
            initial_epoch = cycle*len(y_train_list)*epochs + i*epochs
            end_epoch = initial_epoch + epochs

            # Set contexts explicitly if necessary
            if explicit_contexts[task] is not None:
                model.set_contexts(explicit_contexts[task])

            history = model.fit(x_train, y_train, epochs=end_epoch, initial_epoch=initial_epoch, auto_switch=(explicit_contexts[task] is None), **kwargs)

            # Update the task map
            context_map[task] = model.get_contexts()

            # Track the model history
            history_list.append(history)

        if eval_after_cycle:
            y_test = y_test_list if y_test_list is not None else y_train_list
            eval_list.append(evaluate(model, x_train, y_test, task_map, context_map, display_predictions=False, verbose=0))

        if task_shuffle:
            np.random.shuffle(indices)

    return history_list, eval_list, task_map, context_map


def evaluate(model,
             x, y_list,
             task_map,
             context_map,
             display_predictions=True,
             return_dict=True,
             **kwargs):
    results = []
#     reverse_lookup = np.zeros(len(task_map))
#     for i, task in enumerate(task_map):
#         reverse_lookup[task] = i
    for i, task in enumerate(task_map):
        y = y_list[i]
        contexts = context_map[i]

        model.set_contexts(contexts)
        if display_predictions:
            tf.print(model.predict(x))
        results.append(model.evaluate(x, y, return_dict=return_dict, **kwargs))
    return results
