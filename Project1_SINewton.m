addpath('./utils/');
clear all;
close all;

%% read data
[A,str,label,num,idem,igop] = read_data('all_counties');
[n,dim] = size(A);

%% setup matrix
is = [1,7,5];
[fhandle,XX] = set_up_data(A, str, num, idem, igop, is(1), is(2), is(3));

%% set up optimization problem
for bsz = [n]
[~,dim] = size(XX);
lam = 0.01;
Y = (label*ones(1,dim + 1)).*[XX,ones(n,1)];
w = [-1;-1;1;1];
fun = @(I,w)loss_fun(I,Y,w,lam);
gfun = @(I,w)loss_gfun(I,Y,w,lam);
Hvec = @(I,w,v)loss_Hvec(I,Y,w,v,lam);

[w,f,gnorm] = SINewton(fun,gfun,Hvec,n,w,bsz);

fprintf('w = [%d,%d,%d], b = %d\n',w(1),w(2),w(3),w(4));

plot_plane(fhandle,XX,w)

%%
fsz=16;
figure(4);
hold on;
grid;
niter = length(f);
plot((0:niter-1)',f,'Linewidth',2,'DisplayName',strcat('n=',num2str(bsz)));
set(gca,'Fontsize',fsz);
xlabel('k','Fontsize',fsz);
ylabel('f','Fontsize',fsz);

%%
figure(5);
hold on;
grid;
niter = length(gnorm);
plot((0:niter-1)',gnorm,'Linewidth',2);
set(gca,'Fontsize',fsz);
set(gca,'YScale','log');
xlabel('k','Fontsize',fsz);
ylabel('|| stoch grad f||','Fontsize',fsz);
end
figure(4);
legend;

