num=15;den=[1, 8, 15];
sys = tf(num, den);
t = 0:0.1:10;
y1 = impulse(sys, t);
y2 = 15/2 * (exp(-3*t)-exp(-5*t));
plot(t, y1, t, y2, 'o'),grid;
