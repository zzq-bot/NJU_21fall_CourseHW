function selectedVariables=POSS(n,f,k)
%%  ATTN
%   ATTN: This package is free for academic usage. You can run it at your own risk. For other purposes, please contact Prof. Zhi-Hua Zhou (zhouzh@nju.edu.cn)
%%  ATTN2
%   ATTN2: This package was developed by Mr. Chao Qian (qianc@lamda.nju.edu.cn). For any problem concerning the code, please feel free to contact Mr. Qian.
%%  Some varables used in the code
%   input: 
%      NOTE THAT we use a Boolean string "1001....1100" to represent a subset of variables, where "1" means that the corresponding variable is selected, and "0" means ignoring the variable.
%      n: the total number of variables. 
%      f: a given criterion to be optimized; its input should be a subset of variables, i.e., a Boolean string of length n; its output should be a real value.
%      k: the constraint on the number of selected variables.
%      NOTE THAT we assume that f is to be minimized.
%   ouptut: 
%      selectedVariables: a Boolean string of length n representing the selected variables, the number of which is not larger than k. 


    %initialize the candidate solution set (called "population"): generate a Boolean string with all 0s (called "solution").
    population=zeros(1,n);
    %popSize: record the number of solutions in the population.
    popSize=1;
    %fitness: record the two objective values of a solution.
    fitness=zeros(1,2);
    %the first objective is f; for the special solution 00...00 (i.e., it does not select any variable) and the solutions with the number of selected variables not smaller than 2*k, set its first objective value as inf.  
    fitness(1)=inf;
    %the second objective is the number of selected variables, i.e., the sum of all the bits.
    fitness(2)= sum(population); 

    %repeat to improve the population; the number of iterations is set as 2*e*k^2*n suggested by our theoretical analysis.
    T=round(n*k*k*2*exp(1));
    for i=1:T
        %randomly select a solution from the population and mutate it to generate a new solution.
        offspring=abs(population(randint(1,1,[1,popSize]),:)-randsrc(1,n,[1,0; 1/n,1-1/n]));
        %compute the fitness of the new solution.
        offspringFit=zeros(1,2);
        offspringFit(2)= sum(offspring);        
        if offspringFit(2)==0 || offspringFit(2)>=2*k
            offspringFit(1)=inf;
        else
            offspringFit(1)=f(offspring);
        end

        %use the new solution to update the current population.            
        if sum((fitness(1:popSize,1)<offspringFit(1)).*(fitness(1:popSize,2)<=offspringFit(2)))+sum((fitness(1:popSize,1)<=offspringFit(1)).*(fitness(1:popSize,2)<offspringFit(2)))>0
            continue;
        else
            deleteIndex=((fitness(1:popSize,1)>=offspringFit(1)).*(fitness(1:popSize,2)>=offspringFit(2)))'; 
        end
        %ndelete: record the index of the solutions to be kept.
        ndelete=find(deleteIndex==0);
        population=[population(ndelete,:)',offspring']';
        fitness=[fitness(ndelete,:)',offspringFit']';          
        popSize=length(ndelete)+1;
    end
    
    %select the final solution according to the constraint k on the number of selected variables. 
    temp=find(fitness(:,2)<=k);
    j=max(fitness(temp,2));
    seq=find(fitness(:,2)==j);    
    selectedVariables=population(seq,:);
end