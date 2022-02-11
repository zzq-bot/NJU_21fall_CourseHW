import numpy as np
from utils import *
import math
from PORSS import onepoint_crossover, uniform_crossover, mutation

def POSS_candidate(X, Y, args, draw=False):
    k = args.k
    m,n = X.shape
    P = np.array([np.zeros(n)])
    P_fitness = np.array([[np.inf, 0]])
    Q = np.array([np.zeros(n)])
    Q_fitness = np.array([[np.inf, 0]])

    """PQ = np.concatenate(P, Q)
    PQ_fitness = np.concatenate(P_fitness, Q_fitness)"""

    T = round(2*math.e*n*k*k)#2enk^2
    #T = round(2 * math.e * n * k)
    save_fig_iter = T//5
    prob = [1/(1+math.e**(i-T)) for i in range(T)] #1->1/2
    #prob = [1/2] * T
    for i in range(T):
        print(f"iter {i+1}/{T}:", end=" ")
        #randomly choose
        if np.random.random() < prob[i]:
            #从P中pick
            s = P[np.random.randint(len(P))]
        else:
            s = Q[np.random.randint(len(Q))]
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
        #1。update P
        not_dominate = list(range(len(P))) #idxs that offspring doesn't dominates
        to_add = True
        candidate_Q = []
        candidate_Q_fitness = []
        for idx, f in enumerate(P_fitness):
            #print(P_fitness)
            if f[0] >= offspringFit[0] and f[1] >= offspringFit[1]:
                not_dominate.remove(idx)
                candidate_Q.append(P[idx])
                candidate_Q_fitness.append(f)
            if (f[0] < offspringFit[0] and f[1] <= offspringFit[1]) or (f[0]<=offspringFit[0] and f[1]<offspringFit[1]):#if dominated
                to_add = False
        P = P[not_dominate]
        P_fitness = P_fitness[not_dominate]

        if to_add:
            P = np.concatenate([P, [offspring]])
            P_fitness = np.concatenate([P_fitness, [offspringFit]])


        #2、update Q
        for candidate, candidate_fitness in zip(candidate_Q, candidate_Q_fitness):
            for idx, f in enumerate(Q_fitness):
                if f[1] == candidate_fitness[1]:#它们选择特征的数量一样
                    if f[0] > candidate_fitness[0]:#candidate的mse更小,better
                        Q[idx] = candidate
                        Q_fitness[idx] = candidate_fitness
                    break

        #temp message
        tmp_candidates = np.argwhere(P_fitness[:, 1] <= k).flatten()
        if tmp_candidates is not None:
            tmp_seleced = tmp_candidates[np.argmin(P_fitness[tmp_candidates][:,0])]
            print(f"now popSize:{len(P)}; best_mse:{P_fitness[tmp_seleced][0]}, with num of variables {int(P_fitness[tmp_seleced][1])}")
        """if draw and (i+1) % save_fig_iter == 0:
            save_fig(P_fitness, f'POSS_modified_v1_{args.data} {(i+1)//save_fig_iter}.png')
    if draw:
        save_fig(P_fitness, f'POSS_modified_v1_{args.data} final.png')"""

    candidates = np.argwhere(P_fitness[:,1]<=k).flatten()
    seleced = candidates[np.argmin(P_fitness[candidates][:,0])]
    return P[seleced], P_fitness[seleced]

def PORSS_candidate(X, Y, args, recomb='o'):
    k = args.k
    m, n = X.shape
    P = np.array([np.zeros(n)])
    P_fitness = np.array([[np.inf, 0]])
    Q = np.array([np.zeros(n)])
    Q_fitness = np.array([[np.inf, 0]])

    """PQ = np.concatenate(P, Q)
    PQ_fitness = np.concatenate(P_fitness, Q_fitness)"""

    T = round(2 * math.e * n  * k * k)  # 2enk^2
    # T = round(2 * math.e * n * k)
    save_fig_iter = T // 5
    prob = [1 / (1 + math.e ** (i - T)) for i in range(T)]  # 1->1/2
    # prob = [1/2] * T
    for i in range(T):
        print(f"iter {i + 1}/{T}:", end=" ")
        # randomly choose
        if np.random.random() < prob[i]:
            # 从P中pick
            idx1, idx2 = np.random.choice(len(P), 2, replace=True)
            s1, s2 = P[idx1], P[idx2]
        else:
            if np.random.random() < 1/2:
                idx1, idx2 = np.random.choice(len(Q), 2, replace=True)
                s1, s2 = Q[idx1], Q[idx2]
            else:
                s1 = P[np.random.randint(len(P))]
                s2 = Q[np.random.randint(len(Q))]

        # recombination
        if recomb == 'o':
            c1, c2 = onepoint_crossover(s1, s2)
        elif recomb == 'u':
            c1, c2 = uniform_crossover(s1, s2)
        else:
            print("recomb")
            raise Exception("No such recombination method")
        # mutation
        c1 = mutation(c1)
        c2 = mutation(c2)
        # evaluate
        offspringFit1 = np.array(evaluate(X, Y, c1))
        offspringFit2 = np.array(evaluate(X, Y, c2))

        # update population
        # 1。update P
        for offspring, offspringFit in [[c1, offspringFit1], [c2, offspringFit2]]:
            if offspringFit[1] >= 2 * k:
                continue
            not_dominate = list(range(len(P)))  # idxs that offspring doesn't dominates
            to_add = True
            candidate_Q = []
            candidate_Q_fitness = []
            for idx, f in enumerate(P_fitness):
                # print(P_fitness)
                if f[0] >= offspringFit[0] and f[1] >= offspringFit[1]:
                    not_dominate.remove(idx)
                    candidate_Q.append(P[idx])
                    candidate_Q_fitness.append(f)
                if (f[0] < offspringFit[0] and f[1] <= offspringFit[1]) or (
                        f[0] <= offspringFit[0] and f[1] < offspringFit[1]):  # if dominated
                    to_add = False
            P = P[not_dominate]
            P_fitness = P_fitness[not_dominate]

            if to_add:
                P = np.concatenate([P, [offspring]])
                P_fitness = np.concatenate([P_fitness, [offspringFit]])

            # 2、update Q
            for candidate, candidate_fitness in zip(candidate_Q, candidate_Q_fitness):
                for idx, f in enumerate(Q_fitness):
                    if f[1] == candidate_fitness[1]:  # 它们选择特征的数量一样
                        if f[0] > candidate_fitness[0]:  # candidate的mse更小,better
                            Q[idx] = candidate
                            Q_fitness[idx] = candidate_fitness
                        break

        # temp message
        tmp_candidates = np.argwhere(P_fitness[:, 1] <= k).flatten()
        if tmp_candidates is not None:
            tmp_seleced = tmp_candidates[np.argmin(P_fitness[tmp_candidates][:, 0])]
            print(f"now popSize:{len(P)}; best_mse:{P_fitness[tmp_seleced][0]}, with num of variables {int(P_fitness[tmp_seleced][1])}")
        """if draw and (i+1) % save_fig_iter == 0:
            save_fig(P_fitness, f'POSS_modified_v1_{args.data} {(i+1)//save_fig_iter}.png')
    if draw:
        save_fig(P_fitness, f'POSS_modified_v1_{args.data} final.png')"""

    candidates = np.argwhere(P_fitness[:, 1] <= k).flatten()
    seleced = candidates[np.argmin(P_fitness[candidates][:, 0])]
    return P[seleced], P_fitness[seleced]

def PORSS_candidate_o(X, Y, args, draw=False):
    return PORSS_candidate(X, Y, args, 'o')

def PORSS_candidate_u(X, Y, args, draw=False):
    return PORSS_candidate(X, Y, args, 'u')