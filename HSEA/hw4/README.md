# README

Parameters that can be assigned：

```python
parser.add_argument("--k", default=8, type=int, help="number of selected properties")
parser.add_argument("--data", default="sonar", type=str, help="which dataset to use")
parser.add_argument("--method", default="POSS", type=str, help="which method to use")
parser.add_argument("--repeat", default=1, type=int, help="repeat n times to get mean and std")
parser.add_argument("--popsize", default=40, type=int, help="population size")
parser.add_argument("--epoch", default=400, type=int, help="stopping criteria")
parser.add_argument("--pm", default=-1.0, type=float, help="bit wise mutation, default 1/popSize")
parser.add_argument("--neighbour", default=5, type=int, help="neighbours in consideration in MOEA/D")
```

optional methods：

```python
"POSS": POSS
"NSGA2": NSGA_II  
"MOEADw": MOEA_D, weighted sum
"MOEADt": MOEA_D,Tchbycheff
"PORSSo": PORSS, one-point crossover
"PORSSu": PORSS, uniform crossover
"baseline": GREEDY
"POSS_c": POSS, with candidate,
"PORSS_co": PORSS, one-point crossover with candidate ,
"PORSS_cu": PORSS, uniform crossover with candidate,
"POSS_s": POSS with surrogate function              
```

optional data:

```python
"sonar":sonar_scale
"iono":ionosphere
"svm":svmguide3
"tria":triazines
```



E.g.

Enter the commands：`python main.py --data iono --method POSS --repeat 10`

Run **POSS** algorithm on **ionosphere** dataset with **k=8** **10 times**.

The program will give the mean r^2, mse and std of the result. 

