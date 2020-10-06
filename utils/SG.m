function [w,f,normgrad,ts] = SG(fun,gfun,w,n,bsz,stepsize_toggle,stepsize_param)
%% parameters
kmax = 1e3; % max epoch limit

%% choose stepsize decreasing strategy based on toggle
switch stepsize_toggle
    case 'line_search'
        stepsize_fun = @(Ig,w,s,k) linesearch(Ig,w,s,fun);
    case 'decay'
        stepsize_fun = @(Ig,w,s,k) stepsize_decay(Ig,w,s,fun,k,stepsize_param);
    case 'fixed'
        stepsize_fun = @(Ig,w,s,k) stepsize_fixed(Ig,w,s,fun,stepsize_param);
end

%% setup
I = 1:n;
f = zeros(kmax + 1,1);
ts = zeros(kmax,1);

%% initialization
f(1) = fun(I,w);
normgrad = zeros(kmax,1);
nfail = 0;
nfailmax = 5*ceil(n/bsz);
for k = 1 : kmax
    tic;
    Ig = randperm(n,bsz);
    g = gfun(Ig,w);
    s = -g;
    normgrad(k) = norm(g);

    [a,w,failboo] = stepsize_fun(Ig,w,s,k);
    if failboo
        nfail = nfail+1;
    end
    
    f(k + 1) = fun(I,w);
%     fprintf('k = %d, a = %d, f = %d\n',k,a,f(k+1));
    t = toc;
    ts(k) = t;
    if nfail > nfailmax
        f(k+2:end) = [];
        normgrad(k+1:end) = [];
        ts(k+1:end) = [];
        break;
    end
end
end




