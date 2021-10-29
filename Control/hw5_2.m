num = [1, 10]; den = [1, 15, 0, 0];
sys1 = tf(num, den);
sys = feedback(sys1, 1, -1);
t = 0:0.1:50;
lsim(sys, t, t)