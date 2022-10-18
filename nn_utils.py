import numpy as np

def affine_forward(x, w, b):
    out = None

    # Reshape the input
    x_row = np.expand_dims(x, axis=1).T

    # Perform the affine operation
    out = np.squeeze(np.matmul(x_row, w) + b)
    
    cache = (x_row, w, b)
    return out, cache


def affine_backward(dout, cache):
    x_row, w, b = cache
    dx, dw, db = None, None, None
    
    dout_row = np.expand_dims(dout, axis=0)
    
    dw = np.matmul(x_row.T, dout_row)
    
    dx_row = np.matmul(dout_row, w.T) 
    dx = np.squeeze(dx_row)

    db = dout

    return dx, dw, db


def relu_forward(x):
    out = None
    
    # Compute the mask for positive elements
    mask = (x > 0).astype(np.float64)

    # Apply the mask to calculate ReLU
    out = x * mask

    cache = x
    return out, cache


def relu_backward(dout, cache):
    dx, x = None, cache
    
    # Calculate the mask for positive elements
    mask = (x > 0).astype(np.float64)
    
    # Nullify the gradients corresponding to non-positive elements
    # because they did not contribute to the loss
    dx = dout * mask

    return dx

def softmax_forward(x):
    shifted_logits = x - np.max(x)
    Z = np.sum(np.exp(shifted_logits))
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    
    return probs

def softmax_backward(y, probs):    
    dx = - probs.copy()
    dx[y] += 1

    return dx