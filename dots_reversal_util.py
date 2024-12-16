import numpy as np
import matplotlib.pyplot as plt
import torch

def W_H2H(W_H):
    # W_H ranges -inf < W_H < inf, but H must be 0 < H < 1
    return torch.sigmoid(W_H)

def lam2W_H(lam, refresh_rate):
    return torch.log(lam) - torch.log(refresh_rate - lam)

def W_H2lam(W_H, refresh_rate):
    return W_H2H(W_H) * refresh_rate

def W_k2k(W_k):
    # W_k ranges -inf < W_k < inf, but k must be 0 < k
    return torch.exp(W_k)

def refresh_rate2dt(refresh_rate):
    # msec
    return 1000 / refresh_rate

def calc_prior_expectancy(L, H):
    return L + torch.log((1 - H) / H + torch.exp(-L)) - torch.log((1 - H) / H + torch.exp(L))

def get_correct_responses(df):
    correct_responses = []
    for _, trial in df.iterrows():
        ends_right = int(trial.starts_right)
        ends_right += np.count_nonzero(~np.isnan(trial.values))
        ends_right %= 2
        correct_responses.append(ends_right)
    return torch.tensor(correct_responses)

def get_directions_timeseries(df, refresh_rate):
    """
    1 if dots are moving rightward, 0 if leftward, in each time step.
    """
    directions_right = []
    n_column = len(df.columns)
    i_column_reversal0 = np.argmax(df.columns.values == "reversal0")
    for _, trial in df.iterrows():
        i_reversal = i_column_reversal0
        direction_right = trial.starts_right
        X_trials_inner = []
        i_t = 0
        while (t := i_t * refresh_rate2dt(refresh_rate)) <= trial.trial_duration:
            while i_reversal < n_column and trial.iloc[i_reversal] < t:
                direction_right = 1 - direction_right
                i_reversal += 1

            X_trials_inner.append(direction_right)
            i_t += 1

        directions_right.append(torch.tensor(X_trials_inner))

    return directions_right

def geometric_mean(x: torch.tensor) -> torch.tensor:
    return torch.exp(torch.log(x).mean()).type(torch.float)

def plot(refresh_rate, loss_list, W_H_low_list, W_H_high_list, W_k_list):
    lam_low_list = W_H2lam(torch.tensor(W_H_low_list), refresh_rate)
    lam_high_list = W_H2lam(torch.tensor(W_H_high_list), refresh_rate)
    k_list = W_k2k(torch.tensor(W_k_list))
    fig, ax = plt.subplots(figsize = (4, 2))
    ax.plot(loss_list, label="loss")
    ax.plot(lam_low_list, label="lam_low")
    ax.plot(lam_high_list, label="lam_high")
    ax.plot(k_list, label="k")
    ax.legend()
    plt.show()
    return fig, ax
