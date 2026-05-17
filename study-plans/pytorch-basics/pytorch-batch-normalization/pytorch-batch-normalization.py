import torch

def batch_norm(X, gamma, beta, eps=1e-5):
    """
    Returns: tensor of shape (N, D), the batch-normalized output
    """
    mu=torch.mean(X, dim=0, keepdim=True) # (1,D)
    var=torch.mean((X-mu)**2, dim=0, keepdim=True)
    X_tilde=((X-mu)/(torch.sqrt(var+eps)))
    return gamma*X_tilde+ beta    
