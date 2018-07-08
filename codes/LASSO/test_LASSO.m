clear all
close all
clc
%%
strA = {'australian_label.mat', 'mushrooms_label.mat', 'gisette_label.mat',...
    'covtype_label.mat', 'rcv1_label.mat'};
strB = {'australian_sample.mat', 'mushrooms_sample.mat', 'gisette_sample.mat',...
    'covtype_sample.mat', 'rcv1_sample.mat'};

strF = {'australian', 'mushrooms', 'gisette', 'covtype', 'rcv1'};

i_file = 3;
%% load and scale data
class_name = strA{i_file};
feature_name = strB{i_file};

filename = strF{i_file};

load(['data/', class_name]);
load(['data/', feature_name]);

h = full(h);

% rescale the data
fprintf(sprintf('rescale data...\n'));
itsprint(sprintf('      column %06d...', 1), 1);
for j=1:size(h,2)
    h(:,j) = rescale(h(:,j), -1, 1);
    if mod(j,1e2)==0; itsprint(sprintf('      column %06d...', j), j); end
end
fprintf(sprintf('\nDONE!\n\n'));

h = h(1:2:end, 1:2:end);
l = l(1:2:end);
%% parameters
[m, n] = size(h);

para.m = m;
para.n = n;

para.W = h;
para.y = l;

para.mu = 2e-1;
% if strcmp(filename, 'australian')
%     para.mu = 5e-2;
% elseif strcmp(filename, 'mushrooms')
%     para.mu = 5e-3;
% elseif strcmp(filename, 'gisette')
%     para.mu = 1e-2;
% elseif strcmp(filename, 'covtype')
%     para.mu = 5e-3;
% elseif strcmp(filename, 'rcv1')
%     para.mu = 1e-2;
% end


Li = zeros(m, 1);
for i=1:m
    Wi = para.W(i,:);
    Li(i) = norm(Wi)^2;
end
para.beta_fi = 1 /max(Li);

L = 1/para.beta_fi;

para.tol = 1e-11; % stopping criterion
para.maxits = 6e2*m; % max # of iteration

fprintf(sprintf('      maxits = %09d...\n\n', para.maxits));

ObjF = @(x) para.mu*sum(abs(x)) + norm(para.W*x - para.y)^2/m/2;
GradF = @(x) (para.W)'*(para.W*x - para.y) /m;
iGradF = @(x, i) (para.W(i,:))'*(para.W(i,:)*x - para.y(i));

outputType = 'pdf';
%% SAGA
para.c_gamma = 1/3;

[x1, its1, ek1, fk1, sk1, gk1] = func_SAGA(para, iGradF, ObjF);

fprintf('\n');
%% SAGA, adapt to local Lipschitz const.
para.c_gamma = 1/3;

[x1a, its1a, ek1a, fk1a, sk1a, gk1a] = func_acc_SAGA(para, iGradF, ObjF);

fprintf('\n');
%% SVRG 
para.c_gamma = 1/3;
para.P = m; % # for inner iteration

[x2, its2, ek2, fk2, sk2, gk2] = func_SVRG(para, GradF,iGradF, ObjF);

fprintf('\n');
%% SVRG , adapt to local Lipschitz const.
para.c_gamma = 1/3;
para.P = m; % # for inner iteration

[x2a, its2a, ek2a, fk2a, sk2a, gk2a] = func_acc_SVRG(para, GradF,iGradF, ObjF);

fprintf('\n');
%% step-size
fsol = min([min(fk1), min(fk2)]);

linewidth = 1.5;

axesFontSize = 8;
labelFontSize = 10;
legendFontSize = 10;

resolution = 300; % output resolution
output_size = 300 *[10, 8]; % output size

figure(100), clf;
set(0,'DefaultAxesFontSize', axesFontSize);
set(gcf,'paperunits','centimeters','paperposition',[-0.0 -0.025 output_size/resolution]);
set(gcf,'papersize',output_size/resolution-[0.8 0.4]);


p1 = plot(gk1(1:m:end), 'k', 'LineWidth',linewidth);
hold on,
p1a = plot(gk1a(1:m:end), 'k--', 'LineWidth',linewidth);

p2 = plot(gk2(1:m:end), 'r', 'LineWidth',linewidth);

p2a = plot(gk2a(1:m:end), 'r--', 'LineWidth',linewidth);

grid on;
ax = gca;
ax.GridLineStyle = '--';

axis([1, find(fk2-fsol<1e-12,1), 0 1.5*max(gk1a(end), gk2a(end))]);
% set(gca, 'yTick', [1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e0]);

ylabel({'$\gamma_{k}$'}, 'FontSize', labelFontSize,...
    'FontAngle', 'normal', 'Interpreter', 'latex');
xlabel({'\vspace{-1.0mm}';'$k/m$'}, 'FontSize', labelFontSize,...
    'FontAngle', 'normal', 'Interpreter', 'latex');

lg = legend([p1,p2],...
    sprintf('{SAGA}'), sprintf('{Prox-SVRG}'));
set(lg,'Location', 'Best');
set(lg,'FontSize', 10);
legend('boxoff');
set(lg, 'Interpreter', 'latex');

epsname = sprintf('sagasvrg_lasso_%s_objf.%s', filename, outputType);
if strcmp(outputType, 'png')
    print(epsname, '-dpng');
else
    print(epsname, '-dpdf');
end
%% convergence of \Phi(x_{k}) -\Phi( x_{k-1})
fsol = min([min(fk1), min(fk2)]);

linewidth = 1.5;

axesFontSize = 8;
labelFontSize = 10;
legendFontSize = 10;

resolution = 300; % output resolution
output_size = 300 *[10, 8]; % output size

figure(101), clf;
set(0,'DefaultAxesFontSize', axesFontSize);
set(gcf,'paperunits','centimeters','paperposition',[-0.0 -0.025 output_size/resolution]);
set(gcf,'papersize',output_size/resolution-[0.8 0.4]);


p1 = semilogy(fk1 - fsol, 'k', 'LineWidth',linewidth);
hold on,
p1a = semilogy(fk1a - fsol, 'k--', 'LineWidth',linewidth);

p2 = semilogy(fk2 - fsol, 'r', 'LineWidth',linewidth);

p2a = semilogy(fk2a - fsol, 'r--', 'LineWidth',linewidth);

grid on;
ax = gca;
ax.GridLineStyle = '--';

if strcmp(filename, 'australian')
    axis([1 find(fk1-fsol<1e-10,1) 1e-10 1e-0]);
    set(gca, 'yTick', [1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e0]);
elseif strcmp(filename, 'mushrooms')
    axis([1 find(fk2-fsol<1e-12,1) 1e-16 1e-0]);
    set(gca, 'yTick', [1e-12, 1e-8, 1e-4, 1e0]);
elseif strcmp(filename, 'gisette')
    axis([1 find(fk2-fsol<1e-12,1) 1e-16 1e0]);
    set(gca, 'yTick', [1e-16, 1e-12, 1e-8, 1e-4, 1e0]);
elseif strcmp(filename, 'covtype')
    axis([1 para.maxits/m 1e-10 1e0]);
    set(gca, 'yTick', [1e-3, 1e-2, 1e-1, 1e0]);
elseif strcmp(filename, 'rcv1')
%     axis([1 para.maxits/m 1e-2 3e-1]);
%     set(gca, 'yTick', [1e-2, 1e-1]);
end

ylabel({'$\Phi(x_{k})-\Phi(x^\star)$'}, 'FontSize', labelFontSize,...
    'FontAngle', 'normal', 'Interpreter', 'latex');
xlabel({'\vspace{-1.0mm}';'$k/m$'}, 'FontSize', labelFontSize,...
    'FontAngle', 'normal', 'Interpreter', 'latex');

lg = legend([p1,p2],...
    sprintf('{SAGA}'), sprintf('{Prox-SVRG}'));
set(lg,'Location', 'Best');
set(lg,'FontSize', 10);
legend('boxoff');
set(lg, 'Interpreter', 'latex');

epsname = sprintf('sagasvrg_lasso_%s_objf.%s', filename, outputType);
if strcmp(outputType, 'png')
    print(epsname, '-dpng');
else
    print(epsname, '-dpdf');
end
%% support
fsol = min([min(fk1), min(fk2)]);

linewidth = 1.5;

axesFontSize = 8;
labelFontSize = 10;
legendFontSize = 10;

resolution = 300; % output resolution
output_size = 300 *[10, 8]; % output size

figure(102), clf;
set(0,'DefaultAxesFontSize', axesFontSize);
set(gcf,'paperunits','centimeters','paperposition',[-0.0 0.05 output_size/resolution]);
set(gcf,'papersize',output_size/resolution-[0.8 0.35]);

gap = floor(m/1);

p1 = plot(sk1(1:gap:end), 'k', 'LineWidth',linewidth);
hold on,

p2 = plot(sk2(1:gap:end), 'r--', 'LineWidth',linewidth);

grid on;
ax = gca;
ax.GridLineStyle = '--';

axis([1 its1/gap floor(sk1(end)*3/4) 3*sk1(end)]);
if strcmp(filename, 'australian')
%     axis([1 find(fk1-fsol<1e-10,1) 1e-10 5e-2]);
%     set(gca, 'yTick', [1e-9, 1e-6, 1e-3, 1e0]);
elseif strcmp(filename, 'mushrooms')
    axis([1 find(fk2-fsol<1e-12,1) floor(sk1(end)* 3/4) 2*sk1(end)]);
elseif strcmp(filename, 'gisette')
    axis([1 its2/m floor(sk1(end)* 3/4) 2*sk1(end)]);
elseif strcmp(filename, 'covtype')
    axis([1 para.maxits/m 1e-10 1e0]);
    set(gca, 'yTick', [1e-3, 1e-2, 1e-1, 1e0]);
elseif strcmp(filename, 'rcv1')
%     axis([1 para.maxits/m 1e-2 3e-1]);
%     set(gca, 'yTick', [1e-2, 1e-1]);
end

ylabel({'$|\mathrm{supp}(x_{k})|$'}, 'FontSize', labelFontSize,...
    'FontAngle', 'normal', 'Interpreter', 'latex');
xlabel({'\vspace{-1.2mm}';'$k/m$'}, 'FontSize', labelFontSize,...
    'FontAngle', 'normal', 'Interpreter', 'latex');

lg = legend([p1,p2],...
    sprintf('{SAGA}'), sprintf('{Prox-SVRG}'));
set(lg,'Location', 'Best');
set(lg,'FontSize', 10);
legend('boxoff');
set(lg, 'Interpreter', 'latex');

epsname = sprintf('sagasvrg_lasso_%s_sk.%s', filename, outputType);
if strcmp(outputType, 'png')
    print(epsname, '-dpng');
else
    print(epsname, '-dpdf');
end


