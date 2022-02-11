%a
sys1 = tf([0.5, 2], [1,0]);
sys2 = tf([1], [1, 2, 0]);
sys3 = series(sys1, sys2);
sys = feedback(sys3, 1, -1)

%b
t = 0:0.1:50;
subplot(3, 1, 1), impulse(sys, t)
subplot(3, 1, 2), step(sys, t)
subplot(3, 1, 3), lsim(sys, t, t)