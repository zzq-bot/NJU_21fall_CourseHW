num=21;den=[1, 2, 0];
sys1 = tf(num, den);
sys = feedback(sys1, 1, -1);
t = 0:0.01:5;
step(sys, t)