import numpy as np

def get_alpha_bar(betas) : 

    betas=np.array(betas, dtype=float)
    alpha_bar=np.cumprod(1-betas)
    return alpha_bar.tolist()
    
def reverse_step(x_t, t, epsilon_pred, betas, z=None):
    """
    Returns: np.ndarray x_{t-1} after one reverse diffusion step
    """
    # YOUR CODE HERE
    alpha_t=1-np.array(betas[t-1], dtype=float)
    alpha_bar_t=get_alpha_bar(betas)[t-1]
    beta_t=betas[t-1]
    epsilon_pred=np.array(epsilon_pred, dtype=float)
    coef1=1/(np.sqrt(alpha_t))
    coef2=beta_t/(np.sqrt(1-alpha_bar_t))
    mu=coef1*(x_t - coef2*epsilon_pred)
    
    x_t=np.array(x_t)
    if t>1 and z is not None : 
        # z=np.random.randn(*x_t.shape)
        return np.sqrt(beta_t)*np.array(z)+mu
    else : 
        return mu
 
    