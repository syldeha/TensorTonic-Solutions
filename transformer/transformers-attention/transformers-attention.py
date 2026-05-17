import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    d_model=Q.size(-1)
    attention_weight=F.softmax(torch.matmul(Q,K.transpose(-1,-2))/torch.sqrt(torch.tensor(d_model)), dim=-1)

    return torch.matmul(attention_weight,V)