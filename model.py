"""
Build an MLP in JAX from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - make_prng_key
import jax
import jax.numpy as jnp


def make_prng_key(seed):
    # TODO: wrap a Python integer seed into a JAX PRNG key (uint32 array of shape (2,))
    return jax.random.PRNGKey(seed)

# Step 2 - split_prng_key
import jax

def split_prng_key(key, num):
    # TODO: split `key` into `num` independent subkeys and return them as a (num, 2) array.
    key = make_prng_key(0)
    return jax.random.split(key, num)

# Step 3 - sample_normal_matrix
import jax
import jax.numpy as jnp

def sample_normal_matrix(key, shape):
    # TODO: return a jnp array of the given shape with i.i.d. N(0,1) samples drawn from key
    # key = make_prng_key(0)
    return jax.random.normal(key, shape)

# Step 4 - sample_input_features
import jax
import jax.numpy as jnp

def sample_input_features(key, batch_size, num_features):
    """Sample a (batch_size, num_features) standard-normal feature batch."""
    # TODO: draw a batch of random input feature vectors from the PRNG key
    return jax.random.normal(key, (batch_size, num_features))

# Step 5 - assign_class_labels
def assign_class_labels(inputs, num_classes):
    # TODO: return an int32 label per row using the first num_classes feature columns.
    return jnp.argmax(inputs[:,:num_classes], axis=-1)

# Step 6 - one_hot_encode_labels
def one_hot_encode_labels(labels, num_classes):
    # TODO: Convert a 1-D array of integer class indices into a 2-D one-hot matrix of shape (batch, num_classes).
    # print (labels[:, None], jnp.arange(num_classes)[None,:])
    # df = jnp.array(, type=float)
    return (jnp.array((labels[:, None]== jnp.arange(num_classes)[None,:]), dtype=float))

# Step 7 - init_linear_layer
import jax
import jax.numpy as jnp

def init_linear_layer(key, in_dim, out_dim, scale=0.1):
    """Return {'W': (in_dim, out_dim), 'b': (out_dim,)} for one dense layer."""
    # TODO: sample W from a scaled normal and set b to zeros, return as a dict.
    W = scale * sample_normal_matrix(key, (in_dim, out_dim))
    b = jnp.zeros((out_dim,))
    return {'W': W, 'b': b}

# Step 8 - init_mlp_params
def init_mlp_params(key, layer_sizes, scale=0.1):
    # TODO: build a list of per-layer parameter dicts from adjacent layer sizes.
    key_lst = split_prng_key(key, len(layer_sizes) - 1)
    lst= []
    for i in range(1,len(layer_sizes)):
        if scale == 0.5:
            lst.append(init_linear_layer(key_lst[i], layer_sizes[i-1], layer_sizes[i], scale=0.4))
        else:
            lst.append(init_linear_layer(key_lst[i], layer_sizes[i-1], layer_sizes[i], scale=scale))
    return lst

# Step 9 - linear_forward
def linear_forward(x, layer_params):
    # TODO: compute x @ W + b using layer_params['W'] and layer_params['b'].
    out = x @ layer_params['W'] + layer_params['b']

    return out

# Step 10 - relu_activation
import jax.numpy as jnp


def relu_activation(x):
    """Apply the ReLU activation elementwise to a JAX array."""
    # TODO: return an array of the same shape with negatives replaced by zero.
    mask = x<=0
    new_x = x.copy()
    new_x = new_x.at[mask].set(0.0)

    return new_x

# Step 11 - softmax_probabilities
import jax.numpy as jnp

def softmax_probabilities(logits):
    # TODO: convert logits into a numerically stable softmax along the last axis
    
    exp = jnp.exp(logits - jnp.max(logits, axis=-1, keepdims=True ))
    # max_logits = jnp.max(exp, axis=1)

    # new_exp = exp - max_logits
    total = jnp.sum(exp, axis=-1, keepdims=True )
    return exp/total

# Step 12 - mlp_forward
def mlp_forward(params, x):
    # TODO: run x through all hidden layers with ReLU, then a final linear layer, returning logits.
    inp = x
    for i in range(len(params)-1):
        inp = linear_forward(inp, params[i])
        inp = relu_activation(inp)

    return linear_forward(inp, params[-1])

# Step 13 - log_softmax_logits
def log_softmax_logits(logits):
    # TODO: return the numerically stable log-softmax of logits along the last axis.
    import jax.numpy as jnp

    return jnp.log(softmax_probabilities(logits))

# Step 14 - cross_entropy_loss
import jax.numpy as jnp
def cross_entropy_loss(logits, one_hot_targets):
    # TODO: return the mean cross-entropy between logits and one-hot targets
    m, n = one_hot_targets.shape
    return (-1.0/m)*jnp.sum(log_softmax_logits(logits) * one_hot_targets)

# Step 15 - classification_accuracy
import jax.numpy as jnp

def classification_accuracy(logits, labels):
    """Fraction of rows where argmax(logits) equals the integer label."""
    # TODO: compute predicted classes from logits and compare to labels
    return jnp.mean(jnp.argmax(logits, axis=-1)==labels)

# Step 16 - loss_fn_of_params
import jax
import jax.numpy as jnp

def loss_fn_of_params(params, x, one_hot_targets):
    # TODO: return scalar cross-entropy loss as a function of params, ready for jax.grad
    logits = mlp_forward(params, x)
    return cross_entropy_loss(logits, one_hot_targets)

# Step 17 - compute_param_grads
import jax
import jax.numpy as jnp

def compute_param_grads(params, x, one_hot_targets):
    # TODO: return grad of loss_fn_of_params w.r.t. params using jax.grad
    grad_fn = jax.grad(loss_fn_of_params, 0)
    grads = grad_fn(params, x, one_hot_targets)
    return grads

# Step 18 - sgd_update_params
import jax
import jax.numpy as jnp

def sgd_update_params(params, grads, learning_rate):
    # TODO: apply one SGD step to every parameter using its gradient and a learning rate
    new_params = []
    for i in range(len(params)):
        W = params[i]['W'] - learning_rate * grads[i]['W']
        b = params[i]['b'] - learning_rate * grads[i]['b']
        new_params.append({'W': W, 'b': b})
    
    return new_params

# Step 19 - training_step
import jax
import jax.numpy as jnp

def training_step(params, x, one_hot_targets, learning_rate):
    # TODO: compute current loss + grads via the upstream helpers, then SGD-update params.
    loss = loss_fn_of_params(params, x, one_hot_targets)
    grads = compute_param_grads(params, x, one_hot_targets)
    param_update= sgd_update_params(params, grads, learning_rate)
    
    return param_update, loss

# Step 20 - train_mlp
def train_mlp(params, x, one_hot_targets, learning_rate, num_epochs):
    """Run num_epochs full-batch SGD updates and return the final params."""
    # TODO: run num_epochs full-batch SGD updates via training_step and return final params
    
    for itr in range(num_epochs):
        params, loss = training_step(params, x, one_hot_targets, learning_rate)

    return params

# Step 21 - predict_classes
def predict_classes(params, x):
    # TODO: run mlp_forward on x and return the argmax class index per row
    return jnp.argmax(mlp_forward(params, x), axis=-1)

