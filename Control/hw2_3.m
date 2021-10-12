t = [0:0.01:5];
z = 5;
num = [20/z 20];den = [1 3 20];sys = tf(num, den);
y1 = step(sys, t)

z = 10;
num = [20/z 20];den = [1 3 20];sys = tf(num, den);
y2 = step(sys, t)

z = 15;
num = [20/z 20];den = [1 3 20];sys = tf(num, den);
y3 = step(sys, t)

plot(t, y1, t, y2, t, y3), grid;