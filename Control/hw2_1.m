%(a)
J = 10.8e8; k = 10.8e8; a = 1; b = 8;
num1 = k * [1 a];den1 = [1 b]; sys1 = tf(num1, den1);
num2 = [1]; den2 = [J 0 0]; sys2 = tf(num2, den2); %#ok<*NBRAK>
sys_series = series(sys1, sys2);
sys = feedback(sys_series, [1], -1)

%(b)
t = [0:0.01:100];
A = 10 * pi /180;
sys_A = A * sys;
y = step(sys_A, t);
figure(1)
plot(t, y * 180 / pi), grid;

%(c)
J = 10.8e8 * 0.8
num1 = k * [1 a];den1 = [1 b]; sys1 = tf(num1, den1);
num2 = [1]; den2 = [J 0 0]; sys2 = tf(num2, den2); 
sys_series = series(sys1, sys2);
sys_2 = feedback(sys_series, [1], -1);

sys_A2 = A * sys_2;
y2 = step(sys_A2, t);

J = 10.8e8 * 0.5; k = 10.8e8; a = 1; b = 8;
num1 = k * [1 a];den1 = [1 b]; sys1 = tf(num1, den1);
num2 = [1]; den2 = [J 0 0]; sys2 = tf(num2, den2);
sys_series = series(sys1, sys2);
sys_3 = feedback(sys_series, [1], -1);

sys_A3 = A * sys_3;
y3 = step(sys_A3, t);

figure(2)
plot(t, y * 180 / pi, t, y2 * 180 / pi, t, y3 * 180 / pi), grid;
