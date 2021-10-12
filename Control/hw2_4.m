%(a)
num1 = [1 1];den1=[1 2];sys1 = tf(num1, den1);
num2 = [1];den2=[1 1];sys2 = tf(num2, den2);
sys = feedback(sys1, sys2, -1)

%(b)
pzmap(sys);
p = pole(sys)
z = zero(sys)

%(c)有
minreal(sys)

%(d) 减少系统复杂性，和计算难度