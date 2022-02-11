K=1; tau1 = 2; tau2 = 0.5;
g2 = tf([-10], [1, 10]);
g3 = tf([-1, -6], [1, 3, 6, 0]);
%fast
tau = 0.1;
num = -K * [tau1*tau, tau-2*tau1, -2]; den = [tau2*tau, tau+2*tau2, 2];
g1 = tf(num, den);
sys = series(series(g1, g2), g3)
sys = feedback(sys, [1]);
fast = pole(sys)

%slow
tau = 0.6;
num = -K * [tau1*tau, tau-2*tau1, -2]; den = [tau2*tau, tau+2*tau2, 2];
g1 = tf(num, den);
sys = series(series(g1, g2), g3)
sys = feedback(sys, [1]);
slow = pole(sys)

%bound
tau = 0.2044;
num = -K * [tau1*tau, tau-2*tau1, -2]; den = [tau2*tau, tau+2*tau2, 2];
g1 = tf(num, den);
sys = series(series(g1, g2), g3)
sys = feedback(sys, [1]);
bound = pole(sys)