sys = tf([1], [1, 2, 2, 4, 1, 2]);
pole(sys)
t = 0:0.1:50;
y = step(sys, t);
plot(t, y), grid