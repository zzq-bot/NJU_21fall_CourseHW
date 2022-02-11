------------------------------------------------------------------------------------------
	                   Readme for the POSS Package
	 		       version September 30, 2015
------------------------------------------------------------------------------------------

The package includes the MATLAB code of the POSS (Pareto optimization for subset selection) algorithm, which reformulates subset selection as a bi-objective optimization problem (i.e., optimizing the given criterion and reducing the subset size simultaneously) and then employs a bi-objective evolutionary algorithm [1].

[1] C. Qian, Y. Yu and Z.-H. Zhou. Subset selection by Pareto optimization. In: Advances in Neural Information Processing Systems 28 (NIPS'15), Montreal, Canada, 2015.
 
For POSS, you can call the "POSS" function to run the algorithm. The functions "POSS_MSE" and "POSS_RSS" are specializations of "POSS" for the sparse regression task. For "POSS_MSE", the given criterion is the mean squared error. For "POSS_RSS", the given criterion is the mean squared error + lambda * the l_2 norm regularization. The number of iterations in "POSS" is set as the theoretically suggested value. The isolation function is set as a constant function, and thus can be ignored. 

You will find an example of using this code in the example directory. The example is for sparse regression on the "sonar" data set. You can call the function "main" to run it. 



ATTN: 
- This package is free for academic usage. You can run it at your own risk. For other
  purposes, please contact Prof. Zhi-Hua Zhou (zhouzh@nju.edu.cn).

- This package was developed by Mr. Chao Qian (qianc@lamda.nju.edu.cn). For any
  problem concerning the code, please feel free to contact Mr. Qian.

------------------------------------------------------------------------------------------