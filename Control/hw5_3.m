figure;
t = 0:0.1:10;
i = 1
for omega = [2 1]
    for zeta = [0 0.1 0.2]
        if [omega, zeta] == [2, 0.2]
            continue
        elseif [omega, zeta] == [1, 0.1]
            continue
        else
            sys = tf(omega*omega, [1, 2*zeta*omega, omega*omega]);
            y = impulse(sys, t);
            subplot(2, 2, i), plot(t, y)
            i = i + 1;
        end
    end
end