function [a,w,failboo] = stepsize_fixed(Ig,w,s,fun,fixed_rate)
failboo = false;
f0 = fun(Ig,w);
a = fixed_rate;
w_new = w+a*s;
f_new = fun(Ig,w_new);
if f_new>f0
    failboo = true;
else
    w = w_new;
end
end
