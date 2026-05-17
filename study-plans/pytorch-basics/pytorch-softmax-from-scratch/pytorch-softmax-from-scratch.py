import torch

def softmax(logits):
    """
    Returns: tensor of same shape with softmax probabilities (each row sums to 1)
    """
    max_vals=torch.max(logits, dim=-1, keepdim=True)
    
    max=torch.max(logits)
    num=torch.exp(logits-max)
    return num/num.sum(dim=1, keepdim=True)
