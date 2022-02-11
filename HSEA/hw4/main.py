import argparse
import numpy as np
from utils import get_data
from POSS import POSS
from PORSS import PORSSo, PORSSu
from NSGA_II import NSGA_II
from MOEA_D import MOEA_D_w, MOEA_D_t
from GREEDY import GREEDY
from POSS_candidate import *
from POSS_surrogate import POSS_surrogate
import time
algorithms_dict = {"POSS": POSS,
                   "NSGA2": NSGA_II,
                   "MOEADw": MOEA_D_w,
                   "MOEADt": MOEA_D_t,
                   "PORSSo": PORSSo,
                   "PORSSu": PORSSu,
                   "baseline": GREEDY,
                   "POSS_c": POSS_candidate,
                   "PORSS_co": PORSS_candidate_o,
                   "PORSS_cu": PORSS_candidate_u,
                   "POSS_s": POSS_surrogate
                   }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", default=8, type=int, help="number of selected properties")
    parser.add_argument("--data", default="sonar", type=str, help="which dataset to use")
    parser.add_argument("--method", default="POSS", type=str, help="which method to use")
    #parser.add_argument("--method", default="MOEADw", type=str, help="which method to use")
    parser.add_argument("-seed", default=0, type=int, help="set random seed")
    parser.add_argument("--repeat", default=1, type=int, help="repeat n times to get mean and std")
    parser.add_argument("--popsize", default=40, type=int, help="population size")
    parser.add_argument("--epoch", default=400, type=int, help="stopping criteria")
    parser.add_argument("--pm", default=-1.0, type=float, help="bit wise mutation, default 1/popSize")
    parser.add_argument("--neighbour", default=5, type=int, help="neighbours in consideration in MOEA/D")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    print(args)
    X, Y = get_data(args.data)
    fitness_list = []
    if args.method == "baseline":
        individual, fitness = GREEDY(X, Y, args)
        print("use greedy method baseline")
        print(f"solution:{individual}, r^2:{1-fitness}, mse:{fitness}")
    else:
        algo = algorithms_dict[args.method]
        if args.repeat == 1:
            start = time.time()
            individual, fitness = algo(X, Y, args, True)
            end = time.time()
            print(f"running time: {(end - start) / args.repeat}")
            print(f"one time test, final solution:{individual}, r^2 = {1-fitness[0]} with mse {fitness[0]}")
        else:
            start = time.time()
            for i in range(args.repeat):
                print(f"START {i+1}/{args.repeat}")
                individual, fitness = algo(X, Y, args, False)
                fitness_list.append(fitness[0])
            #print(fitness_list)
            end = time.time()
            print(f"average running time: {(end-start)/args.repeat}")
            print(f"mean r^2 = {1-np.mean(fitness_list)} mse mean = {np.mean(fitness_list)}, std = {np.std(1-np.array(fitness_list))}")

