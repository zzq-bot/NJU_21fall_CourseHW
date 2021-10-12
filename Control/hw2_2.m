%(a)
num1=[1];den1=[1 1];sys1=tf(num1, den1);
num2=[1 0];den2=[1 0 2];sys2=tf(num2, den2);
num3=[4 2];den3=[1 2 1];sys3=tf(num3, den3);
sys4 = feedback(series(sys1, sys2), sys3, -1);

num5=[1];den5=[1 0 0];sys5=tf(num5, den5);
sys6 = feedback(sys5, [50], 1);
sys7 = series(sys4, sys6);

num8 = [1 0 2]; den8 = [1 0 0 14]; sys8 = tf(num8, den8);
sys9 = feedback(sys7, sys8, -1);

sys10 = series([4], sys9);
sys10 = minreal(sys10)


%(b)
pzmap(sys10)


%(c)
p = pole(sys10)
z = zero(sys10)
pzmap(sys8);