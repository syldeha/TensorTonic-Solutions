import numpy as np

def get_alpha_bar(betas):
    """
    Compute cumulative product of (1 - beta).
    Returns list of floats rounded to 6 decimals.
    """
    betas = np.array(betas, dtype=float)

    one_minus_betas = 1 - betas

    cum_prod_one_minus_betas = np.cumprod(one_minus_betas)

    # alpha_bars = np.exp(cum_sum_logs_1_minus_betas)

    return np.round(cum_prod_one_minus_betas,6).tolist()


def forward_diffusion(x_0, t, betas, epsilon):
    """
    Returns: tuple of (np.ndarray x_t, np.ndarray epsilon) with same shape as x_0
    """
    x_0 = np.array(x_0, dtype=float)

    epsilon = np.array(epsilon, dtype=float)

    alpha_bars = get_alpha_bar(betas)

    alpha_bar = alpha_bars[t-1]

    x_t = np.sqrt(alpha_bar) * x_0 + np.sqrt(1 - alpha_bar) * epsilon

    return x_t