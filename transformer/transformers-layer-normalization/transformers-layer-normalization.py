import numpy as np

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Returns: Normalized array of same shape as x
    """
    # Your code here
    mu=np.mean(x, axis=-1, keepdims=True)
    var=np.mean((x-mu)**2 , axis=-1, keepdims=True)
    x_tilde=(x-mu)/np.sqrt(var+eps)

    return gamma*x_tilde+beta