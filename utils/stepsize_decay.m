function [a,w,failboo] = stepsize_decay(Ig,w,s,fun,k,decay_rate)
failboo = false;
f0 = fun(Ig,w);
a = 0.3/(1+decay_rate*k);
w_new = w+a*s;
f_new = fun(Ig,w_new);
if f_new>f0
    failboo = true;
else
    w = w_new;
end
end