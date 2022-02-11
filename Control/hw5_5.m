num = 25; den = [1, 5, 0];
sys1 = tf(num, den);
sys = feedback(sys1, 1, -1);
t = 0:0.001:4;
y = step(sys, t);
plot(t, y), grid


idx = find(y==max(y));
Tp = t(idx) 
Mpt  = (y(idx)-1)/1
text(Tp, y(idx), 'o', 'color', 'r')
text(Tp, y(idx), ['   ', num2str(Tp), ',', num2str(y(idx))], 'color', 'r')

%idx = find(abs(y-1)/1 <= 0.02, 1, 'first')
Ts = 1.62;
idx = find(abs(t-Ts)<1e-3, 1)
Ts = t(idx)
text(Ts, y(idx), 'o', 'color', 'b')
text(Ts, y(idx), ['   ', num2str(Ts), ',', num2str(y(idx))], 'color', 'b')