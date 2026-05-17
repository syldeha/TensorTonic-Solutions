import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    # Your code here
    position=np.arange(seq_length)
    freq=np.zeros((seq_length,d_model)) #(seq, d_model)
    angle=np.exp(((2/d_model)*np.arange((d_model)//2))*np.log(10000))
    angle=position[:,None]/angle[None,:]
    freq[:,0::2]=np.sin(angle)
    freq[:,1::2]=np.cos(angle)

    return freq

    
    