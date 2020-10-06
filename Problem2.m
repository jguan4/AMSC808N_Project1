addpath('./utils/');
clear all;
close all;

%% read data
[A,str,label,num,idem,igop] = read_data('pro_demo');
[n,dim] = size(A);

%% setup matrix
is = [1,7,5];
[fhandle,XX] = set_up_data(A, str, num, idem, igop, is(1), is(2), is(3));

%% set up optimization problem
realizationN = 1000;
[~,dim] = size(XX);
w0 = [-1;-1;1;1];
Y = (label*ones(1,dim + 1)).*[XX,ones(n,1)];

vec = 5;%0:5;
bsz_list = 16*2.^vec;
stepsize_strategies = struct('fixed',[0.5,0.1,0.01],'decay',[0.5,1,2],'line_search',[1]);
lambdas = [0.001,0.01,0.1];

results = struct([]);
counter = 1;
stepsize_names = fieldnames(stepsize_strategies);
for i = 1:length(bsz_list)
    bsz = bsz_list(i);
    for j = 1:length(lambdas)
        lam = lambdas(j);
        fun = @(I,w)loss_fun(I,Y,w,lam);
        gfun = @(I,w)loss_gfun(I,Y,w,lam);
        Hvec = @(I,w,v)loss_Hvec(I,Y,w,v,lam);
        for z = 1:size(stepsize_names,1)
            stepsize_toggle = stepsize_names{z};
            stepsize_params = getfield(stepsize_strategies,stepsize_toggle);
            for zi = 1:length(stepsize_params)
                stepsize_param = stepsize_params(zi);
                fs = zeros(realizationN,1001);
                gnorms = zeros(realizationN,1000);
                tss = zeros(realizationN,1000);
                for ri = 1:realizationN
                    [w,f,gnorm,ts] = SG(fun,gfun,w0,n,bsz,stepsize_toggle,stepsize_param);
                    fs(ri,:) = f';
                    gnorms(ri,:) = gnorm';
                    tss(ri,:) = ts';
                end
                fs_ave = mean(fs,1);
                fs_std = std(fs,1);
                gnorms_ave = mean(gnorms,1);
                gnorms_std = std(gnorms,1);
                tss_ave = mean(tss,1);
                tss_std = std(tss,1);
                
                results(counter).bsz = bsz;
                results(counter).lam = lam;
                results(counter).step = stepsize_toggle;
                results(counter).step_param = stepsize_param;
                results(counter).fs_ave = fs_ave;
                results(counter).fs_std = fs_std;
                results(counter).gnorms_ave = gnorms_ave;
                results(counter).gnorms_std = gnorms_std;
                results(counter).tss_ave = tss_ave;
                results(counter).tss_std = tss_std;
                
                counter = counter+1;
            end
        end
    end
end

save('P2_1.mat','results');

% fprintf('w = [%d,%d,%d], b = %d\n',w(1),w(2),w(3),w(4));

% plot_plane(fhandle,XX,w)
%%
% fsz = 16;
% figure(4);
% hold on;
% grid;
% niter = length(f);
% plot((0:niter-1)',f,'Linewidth',2,'DisplayName',strcat('n=',num2str(bsz)));
% set(gca,'Fontsize',fsz);
% xlabel('k','Fontsize',fsz);
% ylabel('f','Fontsize',fsz);
% 
% %%
% figure(5);
% hold on;
% grid;
% niter = length(gnorm);
% plot((0:niter-1)',gnorm,'Linewidth',2);
% set(gca,'Fontsize',fsz);
% set(gca,'YScale','log');
% xlabel('k','Fontsize',fsz);
% ylabel('|| stoch \nabla f||','Fontsize',fsz);
% 
% figure(4);
% legend;

