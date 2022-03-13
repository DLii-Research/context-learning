import tensorflow as tf

def train_step(model, batch):
    x = batch[0]
    y = batch[1]
    with tf.GradientTape() as tape:
        y_pred = model(x, training=True)
        loss = model.compiled_loss(y, y_pred)
    grads = tape.gradient(loss, model.trainable_weights)
    model.optimizer.apply_gradients(zip(grads, model.trainable_weights))
    context_grads = [grads[i] for i in model.context_gradient_map]
    return context_grads, loss

def make_train_function(eager=False):
    def dist_train_step(strategy, model, batch):
        grads, losses = strategy.run(train_step, args=(model, batch,))

        # Reduce the gradients and add context loss to the model
        grads = strategy.reduce(tf.distribute.ReduceOp.SUM, grads, axis=None)
        model.add_observed_loss(grads)

        return strategy.reduce(tf.distribute.ReduceOp.MEAN, losses, axis=None)
    
    if not eager:
        dist_train_step = tf.function(dist_train_step, experimental_relax_shapes=True)
    
    return dist_train_step