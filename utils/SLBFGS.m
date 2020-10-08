function [x,f,gnorm,ts] =  SLBFGS(fun,gfun,Hvec,x0,n,bszg,bszH,M,stepsize_toggle,stepsize_param)
%% Parameters
tol = 1e-10;
m = 5; % the number of steps to keep in memory
kmax = 1e3;

%% choose stepsize decreasing strategy based on toggle
switch stepsize_toggle
    case 'line_search'
        stepsize_fun = @(Ig,w,s,k) linesearch(Ig,w,s,fun,gfun);
    case 'decay'
        stepsize_fun = @(Ig,w,s,k) stepsize_decay(Ig,w,s,fun,k,stepsize_param);
    case 'fixed'
        stepsize_fun = @(Ig,w,s,k) stepsize_fixed(Ig,w,s,fun,stepsize_param);
end

%% Setup
E = 1:n;
dim = length(x0);
gnorm = [];
f = zeros(kmax + 1,1);

%% choose initial iteration x1 and initialize memory
s = zeros(dim,m);
y = zeros(dim,m);
rho = zeros(1,m);

x = x0;
tic;
Ig = randperm(n,bszg);
g = gfun(Ig,x);

gnorm = [gnorm,norm(g)];
f(1) = fun(E,x);
ts = [];

% first do steepest decend step
[a,~,~] = linesearch(Ig,x,-g,fun,gfun);
xnew = x - a*g;
gnew = gfun(Ig,xnew);
s(:,1) = xnew - x;
y(:,1) = gnew - g;
rho(1) = 1/(s(:,1)'*y(:,1));
x = xnew;
t = toc;
ts = [ts, t];
f(2) = fun(E,xnew);
%% starting iterations
iter = 1;
while iter < kmax
    tic;
    Ig = randperm(n,bszg);
    g = gfun(Ig,x);
    if iter < m
        I = 1 : iter;
        p = finddirection(g,s(:,I),y(:,I),rho(I));
    else
        p = finddirection(g,s,y,rho);
    end
    [a,~,failboo] = stepsize_fun(Ig,x,p,iter);
    if failboo
        p = -g;
        [a,~,~] = stepsize_fun(Ig,x,p,iter);
    end
    step = a*p;
    xnew = x + step;

    if mod(iter,M) == 0
        IH = randperm(n,bszH);
        % newest first
        s = circshift(s,[0,1]); 
        y = circshift(y,[0,1]);
        rho = circshift(rho,[0,1]);
        % inserting at front
        s(:,1) = step;
        y(:,1) = Hvec(IH,x,step);
        rho(1) = 1/(step'*y(:,1));
    end
    t = toc;
    x = xnew;
    nor = norm(g);
    iter = iter + 1;
    gnorm = [gnorm,nor];
    f(iter+1) = fun(E,x);
    ts = [ts, t];
    if nor < tol
        break
    end
end
% fprintf('S L-BFGS: %d iterations, norm(g) = %d\n',iter,nor);
end