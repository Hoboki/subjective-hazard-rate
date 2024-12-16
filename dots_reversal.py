import torch
from torch.distributions.normal import Normal
from dots_reversal_util import W_H2H, W_k2k, calc_prior_expectancy, get_correct_responses, get_directions_timeseries

class DotsReversal:
    def __init__(self, W_H_init, W_k_init, trials, refresh_rate, L_range, dL, responses = None, requires_grad = True) -> None:
        self.W_H = torch.tensor(float(W_H_init), requires_grad=requires_grad)
        self.W_k = torch.tensor(float(W_k_init), requires_grad=requires_grad)
        self.H = W_H2H(self.W_H)
        self.k = W_k2k(self.W_k)
        trial_mask = ~trials.reversal0.isna()
        self.trials = trials[trial_mask].reset_index(drop=True)
        self.coherences = torch.tensor(self.trials.coherence.values, dtype=torch.float)
        self.hazard_rate = torch.tensor(self.trials.hazard_rate.values, dtype=torch.float)
        self.correct_responses = get_correct_responses(self.trials)
        self.responses = self.correct_responses.clone() if responses is None else responses[trial_mask.values]
        self.accuracy = (self.responses == self.correct_responses).sum() / len(self.responses)
        self.n_trial = len(self.trials)
        self.directions_right = get_directions_timeseries(self.trials, refresh_rate)
        self.LogOddsList = torch.arange(-L_range, L_range + 1e-10, dL)
        self.LogOddsTile = torch.tile(self.LogOddsList, (self.LogOddsList.shape[0], 1)).T
        self.IdxLogOddsZero = torch.argmin(torch.square(self.LogOddsList)).item()
        self.DiracDelta = torch.zeros_like(self.LogOddsList)
        self.DiracDelta[self.IdxLogOddsZero] = 1

    def predict(self) -> torch.tensor:
        self.H = W_H2H(self.W_H)
        self.k = W_k2k(self.W_k)
        prior_expectancies = calc_prior_expectancy(self.LogOddsList, self.H)
        pRight_trials = torch.ones(self.n_trial)
        for directions_right, (i_trial, trial) in zip(self.directions_right, self.trials.iterrows()):
            # Initialize belief distribution by Dirac delta
            pL = self.DiracDelta.clone()
            # Make normal distributions for update
            dist_right = Normal(loc = prior_expectancies + self.k * trial.coherence, scale = torch.sqrt(2 * self.k * trial.coherence))
            dist_left = Normal(loc = prior_expectancies - self.k * trial.coherence, scale = torch.sqrt(2 * self.k * trial.coherence))
            dist_tile_right = torch.exp(dist_right.log_prob(self.LogOddsTile))
            dist_tile_left = torch.exp(dist_left.log_prob(self.LogOddsTile))
            for direction_right in directions_right:
                pL_new = (dist_tile_right if direction_right else dist_tile_left) * pL
                pL = pL_new.sum(axis=1) / pL_new.sum() # Update belief distribution

            # Calculate probabilicity that the belief is greater than 0
            pRight = pL[self.IdxLogOddsZero] / 2 + pL[self.IdxLogOddsZero + 1:].sum()
            pRight_trials[i_trial] = pRight

        assert torch.all((0 < pRight_trials) & (pRight_trials < 1))
        return pRight_trials

    def simulate(self) -> torch.tensor:
        self.H = W_H2H(self.W_H)
        self.k = W_k2k(self.W_k)
        prior_expectancies = calc_prior_expectancy(self.LogOddsList, self.H)
        pL_trials_timeseries = []
        for directions_right, (_, trial) in zip(self.directions_right, self.trials.iterrows()):
            # Initialize belief distribution by Dirac delta
            pL = self.DiracDelta.clone()
            # Make normal distributions for update
            dist_right = Normal(loc = prior_expectancies + self.k * trial.coherence, scale = torch.sqrt(2 * self.k * trial.coherence))
            dist_left = Normal(loc = prior_expectancies - self.k * trial.coherence, scale = torch.sqrt(2 * self.k * trial.coherence))
            dist_tile_right = torch.exp(dist_right.log_prob(self.LogOddsTile))
            dist_tile_left = torch.exp(dist_left.log_prob(self.LogOddsTile))
            pL_timeseries = []
            for direction_right in directions_right:
                pL_new = (dist_tile_right if direction_right else dist_tile_left) * pL
                pL = pL_new.sum(axis=1) / pL_new.sum() # Update belief distribution
                pL_timeseries.append(pL)

            pL_timeseries = torch.tensor(pL_timeseries)
            assert torch.all((0 < pL_timeseries) & (pL_timeseries < 1))
            pL_trials_timeseries.append(pL_timeseries)

        return pL_trials_timeseries

    def merge_model(self, model):
        self.W_k = model.W_k

    def make_fake(self, thres_pfc, thres_coherence) -> None:
        torch.manual_seed(0)
        trials_random = torch.rand(len(self.trials))
        self.responses = self.correct_responses.clone()
        incorrect_mask = (self.pfcs < thres_pfc) | (2 * trials_random * self.trials.coherence.values < thres_coherence)
        incorrect_mask = (self.pfcs < thres_pfc)
        self.responses[incorrect_mask] = 1 - self.responses[incorrect_mask]
        self.accuracy = 1 - torch.count_nonzero(incorrect_mask) / self.responses.shape[0]
        return self.responses

    @classmethod
    def loss(cls, predict, target) -> torch.tensor:
        return - (1 - target) * torch.log(1 - predict) - target * torch.log(predict)
