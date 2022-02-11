import numpy as np

from utils import *
import math

def uniform_crossover(s1, s2):
    ret1 = np.zeros_like(s1)
    ret2 = np.zeros_like(s1)
    for i in range(len(ret1)):
        if np.random.rand() < 1/2:
            ret1[i] = s1[i]
            ret2[i] = s2[i]
        else:
            ret1[i] = s2[i]
            ret2[i] = s1[i]
    return ret1, ret2

def onepoint_crossover(s1, s2):
    a = np.random.randint(len(s1))
    ret1 = np.concatenate((s1[:a], s2[a:]), axis=0)
    ret2 = np.concatenate((s2[:a], s1[a:]), axis=0)
    return ret1, ret2

def mutation(s):
    offspring = np.zeros_like(s)
    n = len(s)
    for bit in range(n):
        if np.random.random() < 1/n:
            offspring[bit] = 1-s[bit]
        else:
            offspring[bit] = s[bit]
    return offspring

def PORSS(X, Y, args, draw=False, recomb='o'):
    k = args.k
    m, n = X.shape
    population = np.array([np.zeros(n)])
    popSize = 1
    fitness = np.array([np.array([np.inf, 0])])  # (MSE, num of choosen variables)
    # T = round(2*math.e*n*k*k)#2enk^2
    T = round(2 * math.e * n * k *k)
    save_fig_iter = T // 5

    for i in range(T):
        print(f"iter {i + 1}/{T}:", end=" ")
        # randomly choose
        idx1, idx2 = np.random.choice(len(population), 2, replace=True)
        s1, s2 = population[idx1], population[idx2]
        # recombination
        if recomb == 'o':
            c1, c2 = onepoint_crossover(s1, s2)
        elif recomb == 'u':
            c1, c2 = uniform_crossover(s1, s2)
        else:
            raise Exception("No such recombination method")
        # mutation
        c1 = mutation(c1)
        c2 = mutation(c2)
        # evaluate
        offspringFit1 = np.array(evaluate(X, Y, c1))
        offspringFit2 = np.array(evaluate(X, Y, c2))

        # update population
        for offspring, offspringFit in [[c1, offspringFit1], [c2, offspringFit2]]:
            assert offspringFit.shape[0] == 2 and len(offspringFit.shape) == 1

            if offspringFit[1] >= 2 * k:
                continue
            not_dominate = list(range(popSize))  # idxs that offspring doesn't dominates
            to_add = True
            for idx, f in enumerate(fitness):
                if f[0] >= offspringFit[0] and f[1] >= offspringFit[1]:
                    not_dominate.remove(idx)
                if (f[0] < offspringFit[0] and f[1] <= offspringFit[1]) or (
                        f[0] <= offspringFit[0] and f[1] < offspringFit[1]):  # if dominated
                    to_add = False
            population = population[not_dominate]
            fitness = fitness[not_dominate]

            if to_add:
                population = np.concatenate([population, [offspring]])
                fitness = np.concatenate([fitness, [offspringFit]])
            popSize = len(population)

        # temp message
        tmp_candidates = np.argwhere(fitness[:, 1] <= k).flatten()
        tmp_seleced = tmp_candidates[np.argmin(fitness[tmp_candidates][:, 0])]
        print(
            f"now popSize:{popSize}; best_mse:{fitness[tmp_seleced][0]}, with num of variables {int(fitness[tmp_seleced][1])}")
        if draw and (i + 1) % save_fig_iter == 0:
            save_fig(fitness, f'PORSS_{args.data} {(i + 1) // save_fig_iter}.png')
    if draw:
        save_fig(fitness, f'PORSS_{args.data} final.png')

    candidates = np.argwhere(fitness[:, 1] <= k).flatten()
    seleced = candidates[np.argmin(fitness[candidates][:, 0])]
    return population[seleced], fitness[seleced]

def PORSSo(X, Y, args, draw=False):
    return PORSS(X, Y, args, draw=draw, recomb='o')

def PORSSu(X, Y, args, draw=False):
    return PORSS(X, Y, args, draw=draw, recomb='u')