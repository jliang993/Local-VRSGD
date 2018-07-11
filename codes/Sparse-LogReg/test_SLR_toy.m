clear all
close all
clc
%% generating two clusters of points
m = 128;
d = m* 2;

w_1 = randn(m/2, d)/2;
w_2 = randn(m/2, d)/2;
w_2 = w_2 + 1 + repmat(randn(m/2,1), 1,d);

w = [w_1; w_2];
y = [-ones(m/2,1); ones(m/2,1)];

W = [w, ones(m, 1)];

para.mu = 1 /sqrt(m);
para.beta = 4*m/ norm((W')*W);

L_F = 1/para.beta;

b = zeros(m, 1);
for i=1:m
    Wi = W(i,:);
    b(i) = norm(Wi)^2 /4;
end
para.beta_fi = 1 /max(b);

L = 1/para.beta_fi;

para.tol = 1e-10; % stopping criterion
para.maxits = 1e7; % max # of iteration

para.W = W;
para.y = y;

n = d + 1;
para.m = m;
para.n = n;

GradF = @(x) grad_logistic(x, W, y) /m;
Hess = @(x) hess_logistic(x, W, y);
ProxJ = @(x, t) [wthresh(x(1:end-1), 's', t); x(end)];
iGradF = @(x, i) igrad_logistic(x, i, W, y);

ObjF = @(x) para.mu*sum(abs(x(1:end-1))) + func_logistic(x, para.W, para.y);

outputType = 'pdf';
%% Forward--Backward, finding a high precision solution
para.c_gamma = 2;
[xsol, ~, ek, ~, ~] = func_FB(para, GradF, ProxJ);

fprintf('\n');
%% computing the linear convergence rate of FBS
H = Hess(xsol) /m;

I = (abs(xsol)>0);
kappa = sum(double(I));

PTsol = diag(double(I));

pHp = PTsol * H * PTsol;
s = svd(pHp);

alpha = s(kappa);

PT = diag(double(abs(xsol)>0));
WT = W*PT;
bsol = zeros(m, 1);
for i=1:m
    WTi = WT(i,:);
    bsol(i) = norm(WTi)^2 /4;
end
beta_fi = 1 /max(bsol);
Lsol = max(bsol);
%% SAGA, without acceleration
para.c_gamma = 1/3;

[x1, its1, ek1, fk1, sk1, gk1] = func_SAGA(para, iGradF, ObjF);

fprintf('\n');
%% SAGA, adapt to local Lipschitz const.
para.c_gamma = 1/3;

[x1a, its1a, ek1a, fk1a, sk1a, gk1a, vk1a] = func_acc_SAGA(para, iGradF, ObjF);

fprintf('\n');
%% Prox-SVRG, without acceleration
para.P = m;
para.c_gamma = 1/4;

[x2, its2, ek2, fk2, sk2, gk2] = func_SVRG(para, GradF,iGradF, ObjF);

fprintf('\n');
%% Prox-SVRG, adapt to local Lipschitz const.
para.P = m;
para.c_gamma = 1/4;

[x2a, its2a, ek2a, fk2a, sk2a, gk2a, vk2a] = func_acc_SVRG(para, GradF,iGradF, ObjF);

fprintf('\n');
%% rate estimation of SAGA, Prox-SVRG
P = para.P;

rho_saga = sqrt( 1 - min( 1/(4*m), alpha/(3*L) ) );
rho_svrg = min(1, ( L/ (alpha*L*gk2(1)*P*(1-4*L*gk2(1))) ...
    + (4*L*gk2(1)*(P+1))/((1-4*L*gk2(1))*P) )^(2/P) );

rho_fb_saga = 1 - gk1(1)*alpha;
rho_fb_svrg = 1 - gk2(1)*alpha;

% k1 = floor( find(sk1==sk(end), 1) /m );
% k2 = floor( find(sk2==sk(end), 1) /m );
%% step-size
linewidth = 1.25;

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

axis([1, max(its1, its2)/m, 0 1.1*max(gk1a(end), gk2a(end))]);

ylabel({'$\gamma_{k}$'}, 'FontSize', labelFontSize, 'FontAngle', 'normal', 'Interpreter', 'latex');
xlabel({'\vspace{-0.0mm}';'$k/m$'}, 'FontSize', labelFontSize, 'FontAngle', 'normal', 'Interpreter', 'latex');

lg = legend([p1,p1a, p2, p2a],...
    sprintf('{SAGA}'), sprintf('{acc-SAGA}'),...
    sprintf('{Prox-SVRG}'), sprintf('{acc-Prox-SVRG}'));
set(lg,'Location', 'Best');
set(lg,'FontSize', 10);
legend('boxoff');
set(lg, 'Interpreter', 'latex');

epsname = sprintf('toy_sagasvrg_slr_gamma.%s', outputType);
if strcmp(outputType, 'png')
    print(epsname, '-dpng');
else
    print(epsname, '-dpdf');
end
%% convergence of \Phi(x_{k}) -\Phi(x_{k-1})
fsol = min([min(fk1), min(fk2)]);

linewidth = 1.25;

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

axis([1, max(its1, its2)/m, 1e-14, 1e-1]);


ylabel({'$\Phi(x_{k})-\Phi(x^\star)$'}, 'FontSize', labelFontSize, 'FontAngle', 'normal', 'Interpreter', 'latex');
xlabel({'\vspace{-0.0mm}';'$k/m$'}, 'FontSize', labelFontSize, 'FontAngle', 'normal', 'Interpreter', 'latex');


lg = legend([p1,p1a, p2, p2a],...
    sprintf('{SAGA}'), sprintf('{acc-SAGA}'),...
    sprintf('{Prox-SVRG}'), sprintf('{acc-Prox-SVRG}'));
% set(lg,'Location', 'Best');
set(lg,'FontSize', 10);
legend('boxoff');
set(lg, 'Interpreter', 'latex');

epsname = sprintf('toy_sagasvrg_slr_objf.%s', outputType);
if strcmp(outputType, 'png')
    print(epsname, '-dpng');
else
    print(epsname, '-dpdf');
end
%% support
linewidth = 1.25;

axesFontSize = 8;
labelFontSize = 10;
legendFontSize = 10;

resolution = 300; % output resolution
output_size = 300 *[10, 8]; % output size

figure(103), clf;
set(0,'DefaultAxesFontSize', axesFontSize);
set(gcf,'paperunits','centimeters','paperposition',[-0.0 0.05 output_size/resolution]);
set(gcf,'papersize',output_size/resolution-[0.8 0.35]);

p1 = plot(sk1(1:m:end), 'k', 'LineWidth',linewidth);
hold on,

p2 = plot(sk2(1:m:end), 'r--', 'LineWidth',linewidth);

grid on;
ax = gca;
ax.GridLineStyle = '--';

axis([1 max(its1, its2)/m floor(sk1(end)*3/4) 3*sk1(end)]);

ylabel({'$|\mathrm{supp}(x_{k})|$'}, 'FontSize', labelFontSize, 'FontAngle', 'normal', 'Interpreter', 'latex');
xlabel({'\vspace{-0.0mm}';'$k/m$'}, 'FontSize', labelFontSize, 'FontAngle', 'normal', 'Interpreter', 'latex');


lg = legend([p1,p2],...
    sprintf('{SAGA}'), sprintf('{Prox-SVRG}'));
set(lg,'Location', 'Best');
set(lg,'FontSize', 10);
legend('boxoff');
set(lg, 'Interpreter', 'latex');

epsname = sprintf('toy_sagasvrg_slr_sk.%s', outputType);
if strcmp(outputType, 'png')
    print(epsname, '-dpng');
else
    print(epsname, '-dpdf');
end