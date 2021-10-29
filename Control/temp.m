for T=[1 0.1]
    figure;
    for Zeta = [0 0.2 0.5 0.7 1 10]
        num = 1;
        den = [T*T, 2*Zeta*T, 1];
        sys = tf(num, den);
        step(sys, T*30);
        hold on;
    end;
    grid;
end

