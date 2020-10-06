addpath('./utils/');
clear all;
close all;

%% read data
[A,str,label,num,idem,igop] = read_data('CA');
[n,dim] = size(A);

%% setup matrix
is = [1,7,5];
[fhandle,XX] = set_up_data(A, str, num, idem, igop, is(1), is(2), is(3));

%% set up optimization problem
[n,dim] = size(XX);

% run SINewton to obtain initial guess for ASM
lam = 0.01;
bsz = min(64,n);
Y = (label*ones(1,dim + 1)).*[XX,ones(n,1)];
w = [-1;-1;1;1];
fun = @(I,w)loss_fun(I,Y,w,lam);
gfun = @(I,w)loss_gfun(I,Y,w,lam);
Hvec = @(I,w,v)loss_Hvec(I,Y,w,v,lam);

[w_n,~,~] = SINewton(fun,gfun,Hvec,n,w,bsz);

% soft magins problem
C = 0.5;
D = [eye(dim),zeros(dim, n+1);zeros(n+1, dim), zeros(n+1,n+1)];
E = [zeros(1,dim+1),ones(1,n)];
Aw = [Y, eye(n); zeros(n,dim+1),eye(n)];
b =[ones(n,1);zeros(n,1)];

fun = @(x) x'*D*x/2+C*E*x;
grad_func = @(x) D*x+C*E';
hessian_func = @(x) D;

e = ones(n,1);
xi0 = max(e-Y*w_n,0);
x = [w_n;xi0]; 

% find working set
W = find((Aw*x-b)<1e-14);
% compute w and b using ASM
[wbiter, lm] = ASM(x,grad_func,hessian_func,Aw,b,W);
wb = wbiter(:,end);
w = wb(1:dim+1);

fprintf('w = [%d,%d,%d], b = %d\n',w(1),w(2),w(3),w(4));

plot_plane(fhandle,XX,w_n)
plot_plane(fhandle,XX,w,'cyan')
