import sys
import os
import pandas as pd
import h5py
import torch
from dots_reversal import DotsReversal
from dots_reversal_util import W_H2lam, lam2W_H, W_k2k, geometric_mean

# Parameters
refresh_rate = 60
L_range = 10
dL = 0.4
k = 0.5
n_epoc = 10000
lr =  1 / (3 ** 0.5)
weight_decay = 0
thres_loss = 1e-6
stop_flg_default = 10

# Arguments
i_subject = int(sys.argv[1]) # Accepts only 0 when demo

# Get trials
trials_param = pd.read_csv("trials_param_modulate_hazard_rate.csv")
trials = pd.read_csv("trials_modulate_hazard_rate.csv")
subjects = pd.read_csv("DR_Subject_demo.csv")
subjects = subjects[subjects.condition == "modulate_hazard_rate"]
subjects = subjects[~subjects.finished_at.isna()].reset_index(drop=True)
subject = subjects.iloc[i_subject]
subject_id = subject.id
print(f"{subject_id = }")
responses = pd.read_csv("DR_Response_demo.csv")
responses = responses[responses.subject_id == subject.id].reset_index(drop=True)
responses = torch.tensor((responses.response.values == "j").astype("int8"))

trial_mask = ~trials.reversal0.isna()
trial_mask_low = trial_mask & (trials_param.hazard_rate_state == "easy")
trial_mask_high = trial_mask & (trials_param.hazard_rate_state == "difficult")
trials_low = trials[trial_mask_low].reset_index(drop=True)
trials_high = trials[trial_mask_high].reset_index(drop=True)
responses_low = responses[trial_mask_low]
responses_high = responses[trial_mask_high]
lam_geomean_low = geometric_mean(torch.from_numpy(trials_low.hazard_rate.values.astype("f")))
lam_geomean_high = geometric_mean(torch.from_numpy(trials_high.hazard_rate.values.astype("f")))
print("Geometric Average λ:", lam_geomean_low, lam_geomean_high)
W_H_low = lam2W_H(lam_geomean_low, refresh_rate)
W_H_high = lam2W_H(lam_geomean_high, refresh_rate)
W_k = torch.log(torch.tensor(k))

# Set parameters and optimizer
model_low = DotsReversal(W_H_low, W_k, trials_low, refresh_rate, L_range, dL, responses=responses_low)
model_high = DotsReversal(W_H_high, W_k, trials_high, refresh_rate, L_range, dL, responses=responses_high)
model_high.merge_model(model_low)
print("Accuracy:", model_low.accuracy, model_high.accuracy)
optimizer = torch.optim.SGD([model_low.W_H, model_high.W_H, model_low.W_k], lr = lr, weight_decay = weight_decay)

display_funcs = [lambda: f"{i_epoc = }", lambda: f"{loss = }", lambda: f"{lam_low = }", lambda: f"{lam_high = }", lambda: f"{k = }"]
stop_flg = stop_flg_default
loss_old = 1e10
loss_list = []
H_old = 1e10
W_H_low_list = []
W_H_high_list = []
W_k_list = []
for i_epoc in range(n_epoc):
    loss = 0
    for model in [model_low, model_high]:
        pL_right = model.predict()
        loss = loss + model.loss(pL_right, model.responses).mean()

    loss /= 2
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    loss_list.append(loss.item())
    W_H_low_list.append(model_low.W_H.item())
    W_H_high_list.append(model_high.W_H.item())
    W_k_list.append(model_low.W_k.item())

    lam_low = W_H2lam(model_low.W_H, refresh_rate)
    lam_high = W_H2lam(model_high.W_H, refresh_rate)
    k = W_k2k(model_low.W_k)

    os.system("clear")
    for display_func in display_funcs:
        print(display_func())

    if loss_old - loss.item() < thres_loss:
        stop_flg -= 1

    else:
        stop_flg = stop_flg_default

    loss_old = loss.item()
    if stop_flg == 0:
        break

os.makedirs("_hdf", exist_ok=True)
with h5py.File(f"_hdf/DotsReversal_Pred_Demo.hdf5", "w") as f:
    f.create_dataset("W_H_low", data=model_low.W_H.item())
    f.create_dataset("W_H_high", data=model_high.W_H.item())
    f.create_dataset("W_k", data=model_low.W_k.item())

print("Done.")
