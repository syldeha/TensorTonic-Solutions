import numpy as np

def relu (x):
    m=(x>0)
    return x*m
    

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Apply position-wise feed-forward network.
    """
    # Your code here
    output=relu(x@W1+b1)@W2+b2


    return output