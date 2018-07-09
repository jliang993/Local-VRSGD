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

para.tol = 1e-11; % stopping criterion
para.maxits = 1e5; % max # of iteration

para.W = W;
para.y = y;

n = d + 1;
para.m = m;
para.n = n;

GradF = @(x) grad_logistic(x, W, y) /m;
Hess = @(x) hess_logistic(x, W, y);
ProxJ = @(x, t) [wthresh(x(1:end-1), 's', t); x(end)];
iGradF = @(x, i) igrad_logistic(x, i, W, y);
%% Forward--Backward, finding a high precision solution
para.c_gamma = 2;
[xsol, ~, ek, ~, ~] = func_FB(para, GradF, ProxJ);

fprintf('\n');
%% Forward--Backward
para.c_gamma = 2;
[x, its, dk, sk, gamma] = func_FB_dk(para, GradF, ProxJ, xsol);

fprintf('\n');
%% computing the linear convergence rate of FBS
H = Hess(xsol) /m;

I = (abs(xsol)>0);
kappa = sum(double(I));

PTsol = diag(double(I));

pHp = PTsol * H * PTsol;
s = svd(pHp);

alpha = s(kappa);

PT = diag(double(abs(x)>0));
WT = W*PT;
bsol = zeros(m, 1);
for i=1:m
    WTi = WT(i,:);
    bsol(i) = norm(WTi)^2 /4;
end
beta_fi = 1 /max(bsol);
Lsol = max(bsol);
%% SAGA, without acceleration
para.c_gamma = 1/2;

[x1, its1, dk1, sk1, gamma1] = func_SAGA(para, GradF,iGradF, ProxJ, xsol);

fprintf('\n');
%% SAGA, adapt to local Lipschitz const.
para.c_gamma = 1/2;

[x1a, its1a, dk1a, sk1a, gamma1a] = func_acc_SAGA(para, GradF,iGradF, ProxJ, xsol);

fprintf('\n');
%% Prox-SVRG, without acceleration
para.P = m;
para.c_gamma = 1/3;

[x2, its2,t2, dl2,dk2, sk2, gamma2] = func_SVRG(para, GradF,iGradF, ProxJ, xsol);

fprintf('\n');
%% Prox-SVRG, adapt to local Lipschitz const.
para.P = m;
para.c_gamma = 1/3;

[x2a, its2a,t2a, dl2a,dk2a, sk2a, gamma2a] = func_acc_SVRG(para, GradF,iGradF, ProxJ, xsol);

fprintf('\n');
%% rate estimation of SAGA, Prox-SVRG
P = para.P;

rho_saga = sqrt( 1 - min( 1/(4*m), alpha/(3*L) ) );
rho_svrg = min(1, ( L/ (alpha*L*gamma2*P*(1-4*L*gamma2)) ...
    + (4*L*gamma2*(P+1))/((1-4*L*gamma2)*P) )^(2/P) );

rho_fb_saga = 1 - gamma1*alpha;
rho_fb_svrg = 1 - gamma2*alpha;

k1 = floor( find(sk1==sk(end), 1) /m );
k2 = floor( find(sk2==sk(end), 1) /m );
%% convergence rate of ||x_k-x^\star||
linewidth = 1;

axesFontSize = 8;
labelFontSize = 10;
legendFontSize = 10;

resolution = 300; % output resolution
output_size = 300 *[10, 8]; % output size

figure(101), clf;
set(0,'DefaultAxesFontSize', axesFontSize);
set(gcf,'paperunits','centimeters','paperposition',[-0.0 -0.025 output_size/resolution]);
set(gcf,'papersize',output_size/resolution-[0.8 0.4]);

p1 = semilogy(dk1(1:m:end), 'Color',[0.1,0.1,0.1], 'LineWidth',linewidth);
hold on,

gap = 1;
p2 = semilogy(dk2(1:gap*m:end), 'Color',[0.99,0.1,0.1], 'LineWidth',linewidth);

p1a = semilogy(dk1a(1:m:end), '-.', 'Color',[0.1,0.1,0.1], 'LineWidth',linewidth);
p2a = semilogy(dk2a(1:gap*m:end), '-.', 'Color',[0.99,0.1,0.1], 'LineWidth',linewidth);

p3 = semilogy(k1:1e5, 2*dk1(k1*m)*(rho_fb_saga^m).^(0:1e5-k1), '--', 'Color',[0.1,0.1,0.1], 'LineWidth',linewidth);
p4 = semilogy(k2:1e5, 2*dk2(k2*m)*(rho_fb_svrg^(gap*m)).^(0:1e5-k2), '--', 'Color',[0.99,0.1,0.1], 'LineWidth',linewidth);


grid on;
ax = gca;
ax.GridLineStyle = '--';

axis([1 max([its1/m, t2/m])-1.5e3 1e-8 1e0]);
set(gca, 'yTick', [1e-8, 1e-5, 1e-2, 1e1]);

ylabel({'$\|x_{k}-x^\star\|$'}, 'FontSize', labelFontSize,...
    'FontAngle', 'normal', 'Interpreter', 'latex');
xlabel({'\vspace{-1.0mm}';'$k/m$'}, 'FontSize', labelFontSize,...
    'FontAngle', 'normal', 'Interpreter', 'latex');

lg = legend([p1,p3, p1a, p2,p4,p2a],...
    sprintf('{SAGA}, observation'),...
    sprintf('{SAGA}, $\\rho_{_\\mathrm{FBS}}^{m}$'),...
    sprintf('{SAGA}, acceleration'),...
    sprintf('{Prox-SVRG}, observation'),...
    sprintf('{Prox-SVRG}, $\\rho_{_\\mathrm{FBS}}^{m}$'),...
    sprintf('{Prox-SVRG}, acceleration'));
set(lg,'Location', 'Best');
set(lg,'FontSize', 8);
legend('boxoff');
set(lg, 'Interpreter', 'latex');

pos = get(lg, 'Position');
set(lg, 'Position', [pos(1)-0.07, pos(2)-0.04, pos(3)-0.08, pos(4)-0.065]);
pos_ = get(lg, 'Position');


pdfname = sprintf('LogisticRegression_saga_svrg_rate.png');
print(pdfname, '-dpng');
%% support of x_k
linewidth = 1;

axesFontSize = 8;
labelFontSize = 10;
legendFontSize = 10;

resolution = 300; % output resolution
output_size = 300 *[10, 8]; % output size

figure(102), clf;
set(0,'DefaultAxesFontSize', axesFontSize);
set(gcf,'paperunits','centimeters','paperposition',[-0.0 -0.02 output_size/resolution]);
set(gcf,'papersize',output_size/resolution-[0.8 0.45]);

p1 = plot(sk1(1:m:end), 'Color',[0.1,0.1,0.1], 'LineWidth',linewidth);
hold on,
p2 = plot(sk2(1:m:end), 'Color',[0.99,0.1,0.1], 'LineWidth',linewidth);


grid on;
ax = gca;
ax.GridLineStyle = '--';

axis([1 max([its1, its2])/m sk(end)-2 sk(end)*2]);

ylabel({'$|$supp($x_{k}$)$|$'}, 'FontSize', labelFontSize,...
    'FontAngle', 'normal', 'Interpreter', 'latex');
xlb = xlabel({'$$k/m$$';'\vspace{-0.2cm}'}, 'FontSize', labelFontSize, 'FontAngle', 'normal', 'Interpreter', 'latex');
set(xlb, 'Units', 'Normalized', 'Position', [1/2, -0.08, 0]);

lg = legend([p1, p2], sprintf('SAGA'), sprintf('{Prox-SVRG}'));
set(lg,'Location', 'Best');
set(lg,'FontSize', 9);
legend('boxoff');
set(lg, 'Interpreter', 'latex');


pdfname = sprintf('LogisticRegression_saga_svrg_support.png');
print(pdfname, '-dpng');