import numpy as np
import random
from utils import *
from NSGA_II import mutation


def normalize(X, mean, std):
    return (X-mean) / max(std, 1e-9)

def repair(s, k):
    if np.sum(s) > 2*k:
        idxs = np.argwhere(s==1).flatten()
        if np.random.random() < 0.1:
            choosen_idx = np.random.choice(idxs, k, replace=True)
        else:
            choosen_idx = np.random.choice(idxs, round(2*k), replace=True)
        choosen_idx = np.unique(choosen_idx)
        s = np.zeros_like(s)
        s[choosen_idx] = 1
    return s

def evaluate_target(X, Y, population):
    targets = np.zeros((population.shape[0], 2))
    for i in range(population.shape[0]):
        f1, f2 = evaluate(X, Y, population[i])
        if f1 == np.inf:
            f1 = 1000000
        targets[i][0], targets[i][1] = f1, f2
    return targets


def evaluate_fitness_w(targets, lambdas):
    targets[:, 0] = normalize(targets[:, 0], np.mean(targets[:, 0]), np.std(targets[:, 0]))
    targets[:, 1] = normalize(targets[:, 1], np.mean(targets[:, 1]), np.std(targets[:, 1]))
    fitness = lambdas[:, 0]*targets[:, 0] + lambdas[:, 1] * np.abs(targets[:, 1])
    return fitness

def Tchebycheff(lamb, target, z):
    assert target[0]!=np.inf
    return np.max(lamb * np.abs(target - z))

def evaluate_fitness_t(targets, lambdas, z):
    """targets[:, 0] = normalize(targets[:, 0], np.mean(targets[:, 0]), np.std(targets[:, 0]))
    targets[:, 1] = normalize(targets[:, 1], np.mean(targets[:, 1]), np.std(targets[:, 1]))"""
    fitness = np.zeros(targets.shape[0])
    for i in range(len(fitness)):
        fitness[i] = Tchebycheff(lambdas[i], targets[i], z)
    return fitness

def get_neighbours(lambdas, neighbour_size):
    ret = []#ret[i]: i's neighbours idxes
    for i in range(len(lambdas)):
        dis = [np.linalg.norm(lambdas[i] - lambdas[j]) for j in range(len(lambdas))]
        #print(dis)
        ids = np.argpartition(dis, neighbour_size)
        tmp = ids[:neighbour_size]
        ret.append(tmp)
    return ret

def uniform_crossover(s1, s2):
    ret = np.zeros_like(s1)
    for i in range(len(ret)):
        if np.random.rand() < 1/2:
            ret[i] = s1[i]
        else:
            ret[i] = s2[i]
    return ret

def parent_selection(population):
    return population[random.sample(list(range(population.shape[0])),1)[0]]

def dominate(p, q):
    #print(p, q)
    if (p[0]<=q[0] and p[1]<p[1]) or (p[0]<q[0] and p[1]<=q[1]):
        return True

def init_EP(population, targets):
    """当前population中的非支配解"""
    ret = []
    ret_target = []
    N = targets.shape[0]
    for i in np.arange(N):
        dominated = 0
        for j in np.arange(N):
            if dominate(targets[j], targets[i]):
                dominated += 1
                break
        if dominated == 0:
            ret.append(population[i])
            ret_target.append(targets[i])
    return np.array(ret), np.array(ret_target)

def update_EP(c, target_c, EP, EP_target):
    left_idx = list(range(len(EP)))
    assert len(EP_target.shape) >1 , (EP_target.shape, EP_target)
    assert  len(EP) == len(EP_target)
    dominated = 0
    for i in range(len(EP)):
        if (c == EP[i]).all():
            return EP, EP_target
        if dominate(target_c, EP_target[i]):
            left_idx.remove(i)
        if dominate(EP_target[i], target_c):
            dominated += 1
    EP = EP[left_idx]
    EP_target = EP_target[left_idx]
    if dominated == 0:
        EP = np.concatenate((EP, [c]))
        EP_target = np.concatenate((EP_target, [target_c]))
    """EP = np.unique(EP, axis=0)
    EP_target = evaluate_target(X, Y, EP)"""
    assert len(EP) == len(EP_target)
    return EP, EP_target

def MOEA_D_w(X, Y, args, draw=False):
    print("Start MODE/D")
    k = args.k
    m, n = X.shape
    N = args.popsize
    pm = args.pm
    neighbour_size = args.neighbour
    lambdas = np.random.random((N, 2))
    for i in range(N):
        lambdas[i][0] = (N-i-1)/(N-1)
        lambdas[i][1] = i/(N-1)
    neighbours = get_neighbours(lambdas, neighbour_size)
    population  = np.random.randint(0,2,(N, n))
    for i in range(len(population)):
        population[i] = repair(population[i], k)
    population_target = evaluate_target(X, Y, population)
    EP, EP_target = init_EP(population, population_target)
    population_fitness = evaluate_fitness_w(population_target, lambdas)

    save_fig_iter = args.epoch // 5
    for t in range(args.epoch):
        #print(f"iter {t + 1}/{args.epoch}:")
        # parent selections and offspring generation
        for i in range(N):
            p1 = parent_selection(population[neighbours[i]])
            p2 = parent_selection(population[neighbours[i]])
            p1, p2 = mutation(p1, pm), mutation(p2, pm)
            c = uniform_crossover(p1, p2)
            c = repair(c, k)
            target_c = np.array(evaluate(X, Y, c))
            if target_c[0] == np.inf:
                target_c[0] = 1000000
            for j in range(neighbour_size):
                neighbour = neighbours[i][j]
                #ori_fitness = population_fitness[neighbour]
                f1 = normalize(target_c[0], np.mean(population[:, 0]), np.std(population_target[:, 0]))
                f2 = normalize(target_c[1], np.mean(population[:, 1]), np.std(population_target[:, 1]))
                candidate_f = lambdas[neighbour][0] * f1 + lambdas[neighbour][1] * f2
                if candidate_f <= population_fitness[neighbour]:
                    population[neighbour] = c
                    population_fitness[neighbour] = candidate_f
                    population_target[neighbour] = target_c
            EP, EP_target = update_EP(c, target_c, EP, EP_target)
            assert len(EP) <= 2 * k + 2, print(EP_target)
        """candidates = np.argwhere(EP_target[:, 1] <= k).flatten()
        if len(candidates) > 0:
            seleced = candidates[np.argmin(EP_target[candidates][:, 0])]
            print(EP_target[seleced])"""
        if draw and (t+1)%save_fig_iter == 0:
            save_fig(EP_target, f'MOEA_D_{args.data} {(t + 1) // save_fig_iter}.png')

    candidates = np.argwhere(EP_target[:, 1] <= k).flatten()
    seleced = candidates[np.argmin(EP_target[candidates][:, 0])]
    return EP[seleced], EP_target[seleced]

def MOEA_D_t(X, Y, args, draw=False):
    print("Start MODE/D")
    k = args.k
    m, n = X.shape
    N = args.popsize
    pm = args.pm
    neighbour_size = args.neighbour
    lambdas = np.random.random((N, 2))
    for i in range(N):
        lambdas[i][0] = (N-i-1)/(N-1)
        lambdas[i][1] = i/(N-1)
    neighbours = get_neighbours(lambdas, neighbour_size)
    population  = np.random.randint(0,2,(N, n))
    z = np.array([0, 0])
    for i in range(len(population)):
        population[i] = repair(population[i], k)
    population_target = evaluate_target(X, Y, population)
    EP, EP_target = init_EP(population, population_target)
    population_fitness = evaluate_fitness_t(population_target, lambdas, z)

    for t in range(args.epoch):
        #print(f"iter {t + 1}/{args.epoch}:")
        for i in range(N):
            p1 = parent_selection(population[neighbours[i]])
            assert len(p1.shape)==1, p1.shape
            p2 = parent_selection(population[neighbours[i]])
            p1, p2 = mutation(p1, pm), mutation(p2, pm)
            c = uniform_crossover(p1, p2)
            c = repair(c, k)
            target_c = np.array(evaluate(X, Y, c))
            if target_c[0] == np.inf:
                target_c[0] = 1000000
            for j in range(neighbour_size):
                neighbour = neighbours[i][j]
                """f1 = normalize(target_c[0], np.mean(population[:, 0]), np.std(population_target[:, 0]))
                f2 = normalize(target_c[1], np.mean(population[:, 1]), np.std(population_target[:, 1]))"""
                candidate_f = Tchebycheff(lambdas[neighbour], target_c, z)
                if candidate_f <= population_fitness[neighbour]:
                    population[neighbour] = c
                    population_fitness[neighbour] = candidate_f
                    population_target[neighbour] = target_c
            EP, EP_target = update_EP(c, target_c, EP, EP_target)
            assert len(EP) <= 2*k+2, print(EP_target)
        """candidates = np.argwhere(EP_target[:, 1] <= k).flatten()
        if len(candidates) > 0:
            seleced = candidates[np.argmin(EP_target[candidates][:, 0])]
            print(EP_target[seleced])"""

    candidates = np.argwhere(EP_target[:, 1] <= k).flatten()
    seleced = candidates[np.argmin(EP_target[candidates][:, 0])]
    return EP[seleced], EP_target[seleced]
