import numpy as np
from utils import *
import math


def POSS(X, Y, args, draw=False):
    k = args.k
    m,n = X.shape
    population = np.array([np.zeros(n)])
    popSize = 1
    fitness = np.array([np.array([np.inf, 0])])#(MSE, num of choosen variables)
    T = round(2*math.e*n*k*k)#2enk^2
    #T = round(2 * math.e * n * k)
    save_fig_iter = T//5

    for i in range(T):
        #print(f"iter {i+1}/{T}:", end=" ")
        #randomly choose
        s = population[np.random.randint(len(population))]
        #mutation
        offspring = np.zeros(n)
        for bit in range(n):
            if np.random.random() < 1/n:
                offspring[bit] = 1 - s[bit]
            else:
                offspring[bit] = s[bit]
        if np.sum(offspring) >= 2*k:
            continue
        #evaluate
        f1, f2 = evaluate(X, Y, offspring)
        offspringFit = np.array([f1, f2])

        #update population
        not_dominate = list(range(popSize)) #idxs that offspring doesn't dominates
        to_add = True
        for idx, f in enumerate(fitness):
            if f[0] >= offspringFit[0] and f[1] >= offspringFit[1]:
                not_dominate.remove(idx)
            if (f[0] < offspringFit[0] and f[1] <= offspringFit[1]) or (f[0]<=offspringFit[0] and f[1]<offspringFit[1]):#if dominated
                to_add = False
        population = population[not_dominate]
        fitness = fitness[not_dominate]

        if to_add:
            population = np.concatenate([population, [offspring]])
            fitness = np.concatenate([fitness, [offspringFit]])


        popSize = len(population)

        #temp message
        """tmp_candidates = np.argwhere(fitness[:, 1] <= k).flatten()
        tmp_seleced = tmp_candidates[np.argmin(fitness[tmp_candidates][:,0])]
        print(f"now popSize:{popSize}; best_mse:{fitness[tmp_seleced][0]}, with num of variables {int(fitness[tmp_seleced][1])}")"""
        if draw and (i+1) % save_fig_iter == 0:
            save_fig(fitness, f'POSS_{args.data} {(i+1)//save_fig_iter}.png')
    if draw:
        save_fig(fitness, f'POSS_{args.data} final.png')

    candidates = np.argwhere(fitness[:,1]<=k).flatten()
    seleced = candidates[np.argmin(fitness[candidates][:,0])]
    return population[seleced], fitness[seleced]