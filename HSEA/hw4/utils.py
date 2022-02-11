from libsvm.svmutil import svm_read_problem
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt


dataset_dict = {"sonar": "./data/sonar_scale.txt",
                "iono":"./data/ionosphere_scale.txt",
                "svm":"./data/svmguide3.txt",
                "tria":"./data/triazines_scale.txt"}

def get_data(data):
    if data not in dataset_dict:
        raise Exception("no such dataset")
    else:
        file = dataset_dict[data]
        Y, X = svm_read_problem(file)
        d = list(X[0].keys())[-1]
        # full
        for x in X:
            to_fill = []
            if len(x.keys()) < d:
                for i in range(1, d+1):
                    if i not in x.keys():
                        to_fill.append(i)
            for idx in to_fill:
                x[idx] = 0.0
        X = np.array([list(X[i].values()) for i in range(len(Y))], dtype=np.float32)
        Y = np.array(Y, dtype=np.float32)
        #normalization
        if (np.std(X,axis=0)==0).any():
            X = (X-np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-9)
        else:
            (X - np.mean(X, axis=0)) / (np.std(X, axis=0))
        if (np.std(Y,axis=0)==0).any():
            Y = (Y - np.mean(Y, axis=0)) / (np.std(Y, axis=0) + 1e-9)
        else:
            Y = (Y - np.mean(Y, axis=0)) / np.std(Y, axis=0)
        return X, Y

def evaluate(X, Y, individual):
    if np.sum(individual)==0:
        return np.inf, 0
    choosen_idxs = np.argwhere(individual==1).flatten()
    #print(choosen_idxs)
    X = X[:,choosen_idxs]
    """print(X)
    print(individual)"""
    #print(X.shape, Y.shape)
    lr = linear_model.LinearRegression()
    lr.fit(X, Y)
    #print(lr.score(X, Y))
    y = lr.predict(X)
    #print(y.shape, Y.shape)
    return np.sum((y-Y)*(y-Y))/X.shape[0], len(choosen_idxs)

def save_fig(fitness, name, clear=False):
    if clear:
        plt.cla()
    y = fitness[:,0]
    x = fitness[:,1]
    plt.scatter(x, y, s=10)
    plt.savefig("./pics/"+name)
