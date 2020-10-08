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

vec = 0:6;
bsz_list = 8*2.^vec;
stepsize_strategies = struct('line_search',[1]);
lambdas = [0.01];

results_sg = struct([]);
results_n = struct([]);
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
                fs_sg = zeros(realizationN,1001);
                gnorms_sg = zeros(realizationN,1000);
                tss_sg = zeros(realizationN,1000);
                fs_n = zeros(realizationN,1001);
                gnorms_n = zeros(realizationN,1000);
                tss_n = zeros(realizationN,1000);
                ws_sg = zeros(realizationN,4);
                ws_n = zeros(realizationN,4);
                for ri = 1:realizationN
                    [w,f,gnorm,ts] = SG(fun,gfun,w0,n,bsz,stepsize_toggle,stepsize_param);
                    [w_n,f_n,g_n,ts_n] = SINewton(fun,gfun,Hvec,n,w0,bsz);
                    fs_sg(ri,:) = f';
                    gnorms_sg(ri,:) = gnorm';
                    tss_sg(ri,:) = ts';
                    fs_n(ri,:) = f_n';
                    gnorms_n(ri,:) = g_n';
                    tss_n(ri,:) = ts_n';
                    ws_sg(ri,:) = w';
                    ws_n(ri,:) = w_n';
                end
                fs_ave_sg = mean(fs_sg,1);
                fs_std_sg = std(fs_sg,1);
                gnorms_ave_sg = mean(gnorms_sg,1);
                gnorms_std_sg = std(gnorms_sg,1);
                tss_ave_sg = mean(tss_sg,1);
                tss_std_sg = std(tss_sg,1);
                ws_ave_sg = mean(ws_sg,1);
                
                fs_ave_n = mean(fs_n,1);
                fs_std_n = std(fs_n,1);
                gnorms_ave_n = mean(gnorms_n,1);
                gnorms_std_n = std(gnorms_n,1);
                tss_ave_n = mean(tss_n,1);
                tss_std_n = std(tss_n,1);
                ws_ave_n = mean(ws_n,1);
                
                results_sg(counter).bsz = bsz;
                results_sg(counter).lam = lam;
                results_sg(counter).step = stepsize_toggle;
                results_sg(counter).step_param = stepsize_param;
                results_sg(counter).fs_ave = fs_ave_sg;
                results_sg(counter).fs_std = fs_std_sg;
                results_sg(counter).gnorms_ave = gnorms_ave_sg;
                results_sg(counter).gnorms_std = gnorms_std_sg;
                results_sg(counter).tss_ave = tss_ave_sg;
                results_sg(counter).tss_std = tss_std_sg;
                results_sg(counter).ws_ave_sg = ws_ave_sg;
                
                results_n(counter).bsz = bsz;
                results_n(counter).lam = lam;
                results_n(counter).step = stepsize_toggle;
                results_n(counter).step_param = stepsize_param;
                results_n(counter).fs_ave = fs_ave_n;
                results_n(counter).fs_std = fs_std_n;
                results_n(counter).gnorms_ave = gnorms_ave_n;
                results_n(counter).gnorms_std = gnorms_std_n;
                results_n(counter).tss_ave = tss_ave_n;
                results_n(counter).tss_std = tss_std_n;
                results_n(counter).ws_ave_n = ws_ave_n;
                
                counter = counter+1;
            end
        end
    end
end

save('P3_n.mat','results_n');
save('P3_sg.mat','results_sg');

%% plots
% different g batch size
bsz = bsz_list;
lam = [0.01];
stepsize = struct('line_search',[1]);
inds = find_inds_sg(results_n, bsz, lam, stepsize);
results_plot = results_n(inds);
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

% compare with sg
bszs = [64];
lam = [0.01];
stepsize = struct('line_search',[1]);
results_plot_sg = results_sg(find_inds_sg(results_sg,bszg,lam,stepsize));
results_plot_si = results_n(find_inds_si(results_n,bszg,lam,stepsize));
figure;
plot(1:1001,results_plot_sg(1).fs_ave,'LineWidth',2,'DisplayName','SG');
hold on;
plot(1:1001,results_plot_si(1).fs_ave,'LineWidth',2,'DisplayName','SINewton');
xlabel('Iteration','Fontsize',fsz)
ylabel('f','Fontsize',fsz)
legend;
xlim([1,1000]);
grid on
pbaspect([1.3 1 1])

figure;
plot(1:1000,results_plot_sg(1).gnorms_ave,'LineWidth',2,'DisplayName','SG');
hold on;
plot(1:1000,results_plot_si(1).gnorms_ave,'LineWidth',2,'DisplayName','SINewton');
xlabel('Iteration','Fontsize',fsz)
ylabel('|| stoch grad f||','Fontsize',fsz);
lgd = legend;
xlim([1,1000]);
grid on
pbaspect([1.3 1 1])

a=figure;
plot(1,sum(results_plot_sg(1).tss_ave),'bo','LineWidth',2,'DisplayName','SG');
hold on;
plot(2,sum(results_plot_si(1).tss_ave),'bo','LineWidth',2,'DisplayName','SINewton');
xticks([1,2])
xticklabels({'SLBFGS','SG','SINewton'})
ylabel('runtime (s)','Fontsize',fsz)
xlim([0,3])
grid on
pbaspect([1.3 1 1])

%% functions
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

function inds = find_inds_si(results, bsz, lam, stepsize)
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