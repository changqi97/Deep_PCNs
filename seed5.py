import torch
import numpy
import pcax
import random

import os
import sys
import json

SEEDS = [0, 1, 2, 3, 4, 5, 6]

class run:
    def __init__(self, fn):
        self._seeds = SEEDS

        def wrap_fn(*args, **kwargs):
            best_per_seed = []
            accuracies_per_seed = []
            betst_per_seed5 = []
            accuracies_per_seed5 = []
            for seed in self._seeds:
                torch.manual_seed(seed)
                numpy.random.seed(seed)
                random.seed(seed)
                pcax.RKG.seed(seed)
                if seed == 0:
                    save_model=True
                else:
                    save_model=False
                
                best, best5, accuracies, accuracies5 = fn(*args, save_model=save_model, **kwargs)
                best_per_seed.append(best)
                accuracies_per_seed.append(accuracies)
                betst_per_seed5.append(best5)
                accuracies_per_seed5.append(accuracies5)
            
            return best_per_seed, accuracies_per_seed, betst_per_seed5, accuracies_per_seed5
        self._fn = wrap_fn
        
    def __call__(self, *args, **kwargs):
        best, accuracies, best5, accuracies5 = self._fn(*args, **kwargs)
        
        with open(
            f"{os.path.split(sys.argv[1])[1]}_accuracy.json",
            "w"
        ) as f:
            top5 = numpy.sort(best)[2:]
            top5_ = numpy.sort(best5)[2:]
            json.dump({
                "accuracies": accuracies,
                "avg": numpy.mean(best),
                "std": numpy.std(best),
                "avg5": numpy.mean(top5),
                "std5": numpy.std(top5),
                "accuracies_5": accuracies5,
                "avg_5": numpy.mean(best5),
                "std_5": numpy.std(best5),
                "avg5_5": numpy.mean(top5_),
                "std5_5": numpy.std(top5_)

            }, f, indent=4)