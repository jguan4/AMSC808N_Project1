function [a,w,failboo] = linesearch(Ig,w,s,fun)
failboo = false;
gam = 0.9; %line search decrease factor
jmax = ceil(log(1e-14)/log(gam)); % max # of iterations in line search
eta = 0.5; %in line search
a = 1;
f0 = fun(Ig,w);
aux = eta*(s'*s);
for j = 0 : jmax
    wtry = w + a*s;
    f1 = fun(Ig,wtry);
    if f1 < f0 + a*aux
%         fprintf('Linesearch: j = %d, f1 = %d, f0 = %d\n',j,f1,f0);
        break;
    else
        a = a*gam;
    end
end
if j < jmax
    w = wtry;
else
    failboo = true;
end
end