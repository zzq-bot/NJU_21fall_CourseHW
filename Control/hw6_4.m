p = []
for K = [0:0.1:30]
    G = tf([5], [1, 10, 0]);
    H = tf([2, K], [1, 0]);
    sys = feedback(G, H);
    p = [p pole(sys)];
    i = i+1;
end
plot(real(p), imag(p), 'x'), grid

