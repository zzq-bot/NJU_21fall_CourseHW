import numpy as np
from utils import *
import math

def GREEDY(X, Y, args, draw=False):
    k = args.k
    m,n = X.shape
    best_individual = np.zeros(n)
    best_mse = np.inf
    for _ in range(k):
        tmp_best_ind = None
        tmp_best_mse = np.inf
        for bit in range(n):
            if best_individual[bit] == 0:
                cur_individual = best_individual.copy()
                cur_individual[bit] = 1
                f, _ = evaluate(X, Y, cur_individual)
                if f < tmp_best_mse:
                    tmp_best_mse = f
                    tmp_best_ind = cur_individual
        if tmp_best_mse <= best_mse:
            best_mse = tmp_best_mse
            best_individual = tmp_best_ind
    return best_individual, best_mse