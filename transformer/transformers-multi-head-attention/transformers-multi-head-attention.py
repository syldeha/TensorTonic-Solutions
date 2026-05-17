import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)



def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """
    # Your code here
    batch,seq_len, d_model = Q.shape
    head_dim=d_model//num_heads
    Query=Q@W_q
    key=K@W_k
    value=V@W_v
    Query=Query.reshape(batch, seq_len, num_heads, head_dim).transpose(0,2,1,3)
    key=key.reshape(batch, seq_len, num_heads, head_dim).transpose(0,2,1,3)
    value=value.reshape(batch, seq_len , num_heads, head_dim).transpose(0,2,1,3)

    attention_weight=softmax(np.matmul(Query, key.transpose(0,1,3,2))/np.sqrt(head_dim), axis=-1)
    attention=(attention_weight@value).transpose(0,2,1,3)
    return attention.reshape(batch,seq_len , d_model)@W_o

    
    
    
    