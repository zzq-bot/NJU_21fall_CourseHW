  % We take sparse regression on the sonar data set as an example.
function main()

% load data
[y,X] = libsvmread('svmguide3.txt');
y=full(y);
X=full(X);

% normalization: make all the variables have expectation 0 and variance 1
A = bsxfun(@minus, X, mean(X, 1));
B = bsxfun(@(x,y) x ./ y, A, std(A,1,1));
X=B(:,find(isnan(B(1,:))==0));
A = bsxfun(@minus, y, mean(y, 1));
y = bsxfun(@(x,y) x ./ y, A, std(A,1,1));

% set the size constraint k 
k=8;

% use the POSS_MSE function to select the variables
selectedVariables_MSE=POSS_MSE(X,y,k);

% set the tradeoff parameter lambda between mean squared error and l_2 norm regularization
lambda=0.9615;

% use the POSS_RSS function to select the variables
%selectedVariables_RSS=POSS_RSS(X,y,k,lambda);

end