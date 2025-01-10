import numpy as np
from rich import print as rprint

class CommonParameters:
    def __init__(self, plot_options):
        self.view = plot_options.get('view', 'xz')

        self.seed = plot_options.get('seed', 123456)
        self.hit_survival_prob = plot_options.get('hit_survival_prob', 0.5)

        self.hit_lifetime_ns = plot_options.get('hit_lifetime_ns', 5)


def drop_hits(hit_data, survival_prob, seed):
    rng = np.random.default_rng(seed)
    n_hits_to_drop = rng.binomial(len(hit_data), 1-survival_prob)
    rprint(f'Dropping {n_hits_to_drop} hits out of {len(hit_data)}, survival probability: {survival_prob*100:0.1f}%, hit that survived: {(1-n_hits_to_drop/len(hit_data))*100:0.1f}%')
    indices_to_drop = rng.choice(hit_data.index, n_hits_to_drop, replace=False)
    hit_data.drop(indices_to_drop)
    return hit_data

alphabet = 'abcdefghijklmnopqrstuvwxyz'
