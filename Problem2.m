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

results = struct([]);
counter = 1;

vec = 0:6;
bsz_list = 8*2.^vec;
stepsize_strategies = struct('fixed',[0.5,0.1,0.01],'decay',[0.5,1,2],'line_search',[1]);
lambdas = [0.001,0.01,0.1];

%different batch size
vec = 0:6;
bsz_list = 8*2.^vec;
stepsize_strategies = struct('line_search',[1]);
lambdas = [0.01];
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
                ws = zeros(realizationN,4);
                for ri = 1:realizationN
                    [w,f,gnorm,ts] = SG(fun,gfun,w0,n,bsz,stepsize_toggle,stepsize_param);
                    fs(ri,:) = f';
                    gnorms(ri,:) = gnorm';
                    tss(ri,:) = ts';
                    ws(ri,:) = w';
                end
                fs_ave = mean(fs,1);
                fs_std = std(fs,1);
                gnorms_ave = mean(gnorms,1);
                gnorms_std = std(gnorms,1);
                tss_ave = mean(tss,1);
                tss_std = std(tss,1);
                ws_ave = mean(ws,1);
                
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
                results(counter).ws_ave = ws_ave;
                
                counter = counter+1;
            end
        end
    end
end

%different decreasing strategy
bsz_list = [64];
stepsize_strategies = struct('fixed',[0.5,0.1,0.01],'decay',[0.5,1,2],'line_search',[1]);
lambdas = [0.01];
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
                ws = zeros(realizationN,4);
                for ri = 1:realizationN
                    [w,f,gnorm,ts] = SG(fun,gfun,w0,n,bsz,stepsize_toggle,stepsize_param);
                    fs(ri,:) = f';
                    gnorms(ri,:) = gnorm';
                    tss(ri,:) = ts';
                    ws(ri,:) = w';
                end
                fs_ave = mean(fs,1);
                fs_std = std(fs,1);
                gnorms_ave = mean(gnorms,1);
                gnorms_std = std(gnorms,1);
                tss_ave = mean(tss,1);
                tss_std = std(tss,1);
                ws_ave = mean(ws,1);
                
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
                results(counter).ws_ave = ws_ave;
                
                counter = counter+1;
            end
        end
    end
end

%different decreasing lambdas;
bsz = [64];
lam = lambdas;
stepsize = struct('line_search',[1]);
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
                ws = zeros(realizationN,4);
                for ri = 1:realizationN
                    [w,f,gnorm,ts] = SG(fun,gfun,w0,n,bsz,stepsize_toggle,stepsize_param);
                    fs(ri,:) = f';
                    gnorms(ri,:) = gnorm';
                    tss(ri,:) = ts';
                    ws(ri,:) = w';
                end
                fs_ave = mean(fs,1);
                fs_std = std(fs,1);
                gnorms_ave = mean(gnorms,1);
                gnorms_std = std(gnorms,1);
                tss_ave = mean(tss,1);
                tss_std = std(tss,1);
                ws_ave = mean(ws,1);
                
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
                results(counter).ws_ave = ws_ave;
                
                counter = counter+1;
            end
        end
    end
end

%% plots
% different g batch size
bsz = bsz_list;
lam = [0.01];
stepsize = struct('line_search',[1]);
inds = find_inds_sg(results,bsz, lam, stepsize);
results_plot = results(inds);
figure;
for i = 1:size(results_plot,2)
    legend_name = sprintf('bsz = %d',results_plot(i).bsz);
    ks = 1:size(results_plot(i).fs_ave,2);
    plot(ks,results_plot(i).fs_ave,'LineWidth',2,'DisplayName',legend_name);
    hold on;
end
xlabel('Iteration','Fontsize',fsz)
ylabel('f','Fontsize',fsz)
lgd = legend;
lgd.FontSize = fsz;
xlim([1,1000]);
grid on
pbaspect([1.3 1 1])

figure;
for i = 1:size(results_plot,2)
    legend_name = sprintf('bsz = %d',results_plot(i).bsz);
    ks = 1:size(results_plot(i).gnorms_ave,2);
    plot(ks,results_plot(i).gnorms_ave,'LineWidth',2,'DisplayName',legend_name);
    hold on;
end
xlabel('Iteration','Fontsize',fsz)
ylabel('|| stoch grad f||','Fontsize',fsz);
lgd = legend;
lgd.FontSize = fsz;
xlim([1,1000]);
grid on
pbaspect([1.3 1 1])

a=figure;
for i = 1:size(results_plot,2)
    legend_name = sprintf('bsz = %d',results_plot(i).bsz);
    plot(results_plot(i).bsz,sum(results_plot(i).tss_ave),'bo','LineWidth',2,'DisplayName',legend_name);
    hold on;
end
xlabel('bsz','Fontsize',fsz)
ylabel('runtime (s)','Fontsize',fsz)
xlim([min(bsz_list)-5,max(bsz_list)+5])
grid on
pbaspect([1.3 1 1])

% different lambda
bsz = [64];
lam = lambdas;
stepsize = struct('line_search',[1]);
inds = find_inds_sg(results,bsz, lam, stepsize);
results_plot = results(inds);
figure;
for i = 1:size(results_plot,2)
    legend_name = sprintf('lam = %d',results_plot(i).lam);
    ks = 1:size(results_plot(i).fs_ave,2);
    plot(ks,results_plot(i).fs_ave,'LineWidth',2,'DisplayName',legend_name);
    hold on;
end
xlabel('Iteration','Fontsize',fsz)
ylabel('f','Fontsize',fsz)
lgd = legend;
lgd.FontSize = fsz;
xlim([1,1000]);
grid on
pbaspect([1.3 1 1])

figure;
for i = 1:size(results_plot,2)
    legend_name = sprintf('lam = %d',results_plot(i).lam);
    ks = 1:size(results_plot(i).gnorms_ave,2);
    plot(ks,results_plot(i).gnorms_ave,'LineWidth',2,'DisplayName',legend_name);
    hold on;
end
xlabel('Iteration','Fontsize',fsz)
ylabel('|| stoch grad f||','Fontsize',fsz);
lgd = legend;
lgd.FontSize = fsz;
xlim([1,1000]);
grid on
pbaspect([1.3 1 1])

a=figure;
for i = 1:size(results_plot,2)
    legend_name = sprintf('lam = %d',results_plot(i).lam);
    plot(results_plot(i).lam,sum(results_plot(i).tss_ave),'bo','LineWidth',2,'DisplayName',legend_name);
    hold on;
end
xlabel('lam','Fontsize',fsz)
ylabel('runtime (s)','Fontsize',fsz)
xlim([min(lambdas)-0.1,max(lambdas)+0.1])
grid on
pbaspect([1.3 1 1])

% different stepsize
bsz = [64];
lam = [0.01];
stepsize = stepsize_strategies;
inds = find_inds_sg(results,bsz, lam, stepsize);
results_plot = results(inds);
lgd_names = {'\alpha=0.5','\alpha=0.1','\alpha=0.01','\gamma=0.5','\gamma=1','\gamma=2', 'line search'};
figure;
for i = 1:size(results_plot,2)
    legend_name = lgd_names{i};
    ks = 1:size(results_plot(i).fs_ave,2);
    plot(ks,results_plot(i).fs_ave,'LineWidth',2,'DisplayName',legend_name);
    hold on;
end
xlabel('Iteration','Fontsize',fsz)
ylabel('f','Fontsize',fsz)
lgd = legend;
xlim([1,1000]);
grid on
pbaspect([1.3 1 1])

figure;
for i = 1:size(results_plot,2)
    legend_name = lgd_names{i};
    ks = 1:size(results_plot(i).gnorms_ave,2);
    plot(ks,results_plot(i).gnorms_ave,'LineWidth',2,'DisplayName',legend_name);
    hold on;
end
xlabel('Iteration','Fontsize',fsz)
ylabel('|| stoch grad f||','Fontsize',fsz);
lgd = legend;
xlim([1,1000]);
grid on
pbaspect([1.3 1 1])

a=figure;
for i = 1:size(results_plot,2)
    legend_name = lgd_names{i};
    plot(i,sum(results_plot(i).tss_ave),'bo','LineWidth',2,'DisplayName',legend_name);
    hold on;
end
xticks([1,2,3,4,5,6,7])
xticklabels(lgd_names)
ylabel('runtime (s)','Fontsize',fsz)
xlim([0,8])
xtickangle(45)
grid on
pbaspect([1.3 1 1])

%% function
function inds = find_inds_sg(results, bsz, lam, stepsize)
stepsize_names = fieldnames(stepsize);
dim = size(results,2);
inds = [];
for i = 1:dim
    if ismember(results(i).bsz, bsz) && ismember(results(i).lam, lam) 
        if ismember(results(i).step, stepsize_names)
            params = getfield(stepsize,results(i).step);
            if ismember(results(i).step_param, params)
                inds = [inds,i];
            end
        end
    end
end
end
