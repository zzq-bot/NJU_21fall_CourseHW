import numpy as np
import random
from utils import *

def dominate(p, q):
    if (p[0]<=q[0] and p[1]<p[1]) or (p[0]<q[0] and p[1]<=q[1]):
        return True

def fast_nondominated_sorting(fitness):
    Sorted = [] #[[],[]]
    N = fitness.shape[0]
    S = {}  #i:[j that is dominated by i]
    num_dominated = np.zeros(N) #[i]: num of j that dominates i
    r = 1   #current rank
    F = []  #current rank r set
    for i in np.arange(N):
        S[i] = []
        for j in np.arange(N):
            if dominate(fitness[i], fitness[j]): #i<j
                S[i].append(j)
            if dominate(fitness[j], fitness[i]):
                num_dominated[i] += 1
        if num_dominated[i] == 0:
            fitness[i][2] = r
            F.append(i)
    while F:
        Sorted.append(F)
        Q = []
        for i in F:
            for j in S[i]:
                num_dominated[j] -= 1
                if num_dominated[j] == 0:
                    fitness[j][2] = r+1
                    Q.append(j)
        r += 1
        F=Q
    return Sorted, fitness

def crowding_distance(P, fitness, num):
    n = P.shape[0]
    distances = np.zeros(n)
    #f1
    f1_rank = np.argsort(fitness[:, 0])#升序
    f1_max = np.max(fitness[:, 0])
    f1_min = np.min(fitness[:, 0])
    if f1_max == np.inf or f1_max==np.nan:
        print(fitness)
        assert 0
    assert f1_max == fitness[:,0][f1_rank[-1]]
    distances[f1_rank[0]] = 100000.0
    distances[f1_rank[n-1]] = 100000.0
    for i in range(1, n-1):
        try:
            distances[f1_rank[i]] += (fitness[:, 0][f1_rank[i+1]] - fitness[:, 0][f1_rank[i-1]])/ (f1_max-f1_min)
        except:
            print(fitness[:, 0][f1_rank[i + 1]] - fitness[:, 0][f1_rank[i - 1]])
            print(f1_max - f1_min)
    #f2
    f2_rank = np.argsort(fitness[:, 0])  # 升序
    f2_max = np.max(fitness[:, 0])
    f2_min = np.min(fitness[:, 0])
    assert f2_max == fitness[:, 0][f2_rank[-1]]
    distances[f2_rank[0]] = 100000.0
    distances[f2_rank[n - 1]] = 100000.0
    for i in range(1, n - 1):
        try:
            distances[f2_rank[i]] += (fitness[:, 0][f2_rank[i + 1]] - fitness[:, 0][f2_rank[i - 1]]) / (f2_max - f2_min)
        except:
            print(fitness[:, 0][f2_rank[i + 1]] - fitness[:, 0][f2_rank[i - 1]])
            print(f2_max-f2_min)
    return P[np.argsort(distances)[::-1][:num]], fitness[np.argsort(distances)[::-1][:num]]


def binary_tournament(P, fitness):
    #print(P.shape, fitness.shape)
    competitor_1, competitor_2 = random.sample(list(range(P.shape[0])), 2)
    if fitness[competitor_1][-1] < fitness[competitor_2][-1]:
        return P[competitor_1]
    else:
        return P[competitor_2]

def evaluate_target(X, Y, population):
    fitness = np.zeros((population.shape[0], 3)) #(mse, k, rank) mse,k小->rank小
    for i in range(population.shape[0]):
        f1, f2 = evaluate(X, Y, population[i])
        fitness[i][0] = f1 # min (f1, f2)
        fitness[i][1] = f2
        if fitness[i][0] == np.inf:
            fitness[i][0] = 1000000
    return fitness

def crossover(s1, s2):
    a = np.random.randint(len(s1))
    ret1 = np.concatenate((s1[:a], s2[a:]), axis=0)
    ret2 = np.concatenate((s2[:a], s1[a:]), axis=0)
    return ret1, ret2

def mutation(s, pm):
    for i in range(len(s)):
        if np.random.random() < pm:
            s[i] = 1-s[i]
    return s

def NSGA_II(X, Y, args, draw=False):
    k = args.k
    m, n = X.shape
    N = args.popsize
    pm = args.pm
    if pm == -1:
       pm = 1/n
    #initial population
    population = np.random.randint(0,2,(N, n))
    # evaluate fitness
    population_fitness = evaluate_target(X, Y, population)
    population_fitness = fast_nondominated_sorting(population_fitness)[1]  # (mse, k, rank)
    save_fig_iter = args.epoch // 5
    for t in range(args.epoch):
        #print(f"iter {t + 1}/{args.epoch}:", end=" ")
        #parent selections and offspring generation
        offsprings = []
        while len(offsprings) < N:
            p1 = binary_tournament(population, population_fitness)
            p2 = p1
            #print(p1, p2)
            while (p2 == p1).all():
               p2 = binary_tournament(population, population_fitness)
            c1, c2 = crossover(p1, p2)
            c1, c2 = mutation(c1, pm), mutation(c2, pm)
            offsprings.append(c1)
            offsprings.append(c2)
        offsprings = np.array(offsprings)
        assert len(offsprings) == N

        #surivior selection
        offsprings_fitness = evaluate_target(X, Y, offsprings)
        candidates_population = np.concatenate((population, offsprings), axis=0)
        candidates_fitness = np.concatenate((population_fitness, offsprings_fitness), axis=0)
        Sorted_idxs, candidates_fitness = fast_nondominated_sorting(candidates_fitness)
        next_population = []
        next_population_fitness = []
        cur_rank = 1
        while len(next_population) + len(Sorted_idxs[cur_rank-1]) <= N:
            next_population += list(candidates_population[Sorted_idxs[cur_rank-1]])
            next_population_fitness += list(candidates_fitness[Sorted_idxs[cur_rank-1]])
            cur_rank+=1
        left = N-len(next_population)
        if left > 0: #crowding distance
            choosen_p, choosen_f = crowding_distance(candidates_population[Sorted_idxs[cur_rank-1]],
                                                     candidates_fitness[Sorted_idxs[cur_rank-1]], left)
            next_population += list(choosen_p)
            next_population_fitness += list(choosen_f)

        population = np.array(next_population)
        population_fitness = np.array(next_population_fitness)

        """tmp_candidates = np.argwhere(population_fitness[:, 1] <= k).flatten()
        if len(tmp_candidates)>0:
            #print(tmp_candidates)
            tmp_seleced = tmp_candidates[np.argmin(population_fitness[tmp_candidates][:, 0])]
            print(f"now best_mse:{population_fitness[tmp_seleced]}")
        else:
            print("")"""
        if draw and (t + 1) % save_fig_iter == 0:
            save_fig(population_fitness[:,:2], f'NSGA_II_{args.data} {(t + 1) // save_fig_iter}.png')

    candidates = np.argwhere(population_fitness[:, 1] <= k).flatten()
    seleced = candidates[np.argmin(population_fitness[candidates][:, 0])]
    return population[seleced], population_fitness[seleced]

