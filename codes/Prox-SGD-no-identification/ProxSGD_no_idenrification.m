clear all;
% close all;
clc;
%% problem setup
n = 3; % dimension of x0

xob = zeros(n, 1);
xob(1) = 8;

A = [1, 0, 0; 0, sqrt(2), 0; 0, 0, sqrt(3)];
y = [2; sqrt(2)/3; sqrt(3)/4];

ProxJ = @(x, t) wthresh(x, 's', t);
GradF = @(x) (A')*(A*x - y) /3;
%% parameters
beta = 3 /norm(A)^2;
mu = 1/3;

gamma = 1e0 *beta;
tau = mu*gamma;

maxits = 1e8 + 1;
ToL = 1e-15;

FBS = @(x, g) ProxJ(x-gamma*g, tau);
ProxSGD = @(x, g, gamma) ProxJ(x-gamma*g, mu*gamma);
%% deterministic FBS
fprintf(sprintf('performing Forward--Backward...\n'));

x0 = 1e1* y;

[x1, its1, dk1, sk1] = func_FB(x0, GradF, FBS);

fprintf('\n');
%% Prox-SGD, starting point 1
fprintf(sprintf('performing Prox-SGD...\n'));
x0 = -1e1* (-GradF(x1)/mu);

[x2, its2, ek2, sk2] = func_ProxSGD(x0,y,A, gamma, ProxSGD);

fprintf('\n');
%% Prox-SGD, starting point 2
fprintf(sprintf('performing Prox-SGD...\n'));
x0 = 1e2* x1;

[x3, its3, ek3, sk3] = func_ProxSGD(x0,y,A, gamma, ProxSGD);

fprintf('\n');
%% supp(xk)
axesFontSize = 7;
labelFontSize = 10;
legendFontSize = 9;

resolution = 300; % output resolution
output_size = 300 *[12, 8]; % output size

figure(112), clf;
set(0,'DefaultAxesFontSize', axesFontSize);
set(gcf,'paperunits','centimeters','paperposition',[-0.2 -0.0 output_size/resolution]);
set(gcf,'papersize',output_size/resolution-[0.85 0.4]);

p1 = plot(sk1, 'k', 'linewidth', 1);
hold on;

p2 = plot(sk2(1:1e5:end), 'b', 'linewidth', 0.8);

p3 = plot(sk3(1:1e5:end), 'r', 'linewidth', 0.8);

grid on;
axis([1, max([its1, its2, its3]/1e5), 1/2, 4]);
set(gca, 'yTick', 0:1:4);
set(gca, 'xTick', 0:1e1:max([its1, its2, its3])/1e4);

ylb = ylabel({'$$|$$supp$$(x_{k})|$$';},...
    'FontSize', labelFontSize, 'FontAngle', 'normal', 'Interpreter', 'latex');
set(ylb, 'Units', 'Normalized', 'Position', [-0.06, 0.5, 0]);
xlb = xlabel({'$$k$$';'\vspace{-0.2cm}'}, 'FontSize', labelFontSize, 'FontAngle', 'normal', 'Interpreter', 'latex');
set(xlb, 'Units', 'Normalized', 'Position', [1/2, -0.075, 0]);

lg = legend([p1, p2, p3], 'Forward--Backward', 'Prox-SGD: initial point 1', 'Prox-SGD: initial point 2');
set(lg, 'FontSize', 9);
% set(lg,'Interpreter','latex');
legend('boxoff');

pos = get(lg, 'Position');  
set(lg, 'Position', [pos(1)-0.07, pos(2)-0.0, pos(3)-0.08, pos(4)-0.065]);
pos_ = get(lg, 'Position');

print('Supp-ProxSGD-LASSO.png', '-dpng');