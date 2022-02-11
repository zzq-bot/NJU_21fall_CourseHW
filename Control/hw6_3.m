p = []
for K = [0:0.1:5]
    sys = tf([1],[1,5,K-3,K+1]);
    p = [p pole(sys)];
end
plot(real(p), imag(p), 'x'), grid

%c
K = 4;
sys = tf([1],[1,5,K-3,K+1]);
pole(sys)