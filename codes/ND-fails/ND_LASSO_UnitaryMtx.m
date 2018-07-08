clear all;
close all;
clc;
%% problem setup
n = 16; % dimension of x0
m = n;

A = dctmtx(n);

y = (rand(n, 1)-1/2) /2;
y(5) = 3;
y(11) = -3;

y([1,4,9,14], 1) = 1/2;
y([7,12,16], 1) = -1/2;

f = A*y;
%% parameters
beta = 1;
mu = 1/2;

xsol = wthresh(y, 's', mu);
gsol = y - xsol;
vsol = xsol + gsol/mu;

gamma = 1e-1 *beta;
tau = mu*gamma;

GradF = @(x) (x - y); % use y instead of A'f, the latter is not numerically stable
FBS = @(x, g) wthresh(x-gamma*g, 's', tau);
%% Starting point 1
x0 = 1e2* gsol;

[x1, its1, dk1, sk1] = func_FB(x0,xsol, GradF,FBS);
%% Starting point 2
x0 = 1e3* gsol;

x0(1) = -1e1*x0(1);
x0(4) = -2e2*x0(4);
x0(7) = -3e3*x0(7);
x0(14) = -4e4*x0(14);
x0(16) = -5e5*x0(16);

[x2, its2, dk2, sk2] = func_FB(x0,xsol, GradF,FBS);
%% Starting point 3
x0 = -1e5* gsol;

[x3, its3, dk3, sk3] = func_FB(x0,xsol, GradF,FBS);
%% supp(xk)
axesFontSize = 7;
labelFontSize = 10;
legendFontSize = 9;

resolution = 300; % output resolution
output_size = 300 *[10, 8]; % output size

%%%%%% relative error

figure(112), clf;
set(0,'DefaultAxesFontSize', axesFontSize);
set(gcf,'paperunits','centimeters','paperposition',[-0.2 -0.2 output_size/resolution]);
set(gcf,'papersize',output_size/resolution-[0.85 0.6]);

p1 = plot(sk1, 'r', 'linewidth', 1);

hold on;

p2 = plot(sk2, 'b', 'linewidth', 1);

p3 = plot(sk3, 'k', 'linewidth', 1);

grid on;
axis([1, max([its1, its2, its3]), 0, 17]);
set(gca, 'yTick', 0:2:16);
set(gca, 'xTick', 0:100:max([its1, its2, its3]));

ylb = ylabel({'$$|$$supp$$(x_{k})|$$';},...
    'FontSize', labelFontSize, 'FontAngle', 'normal', 'Interpreter', 'latex');
set(ylb, 'Units', 'Normalized', 'Position', [-0.06, 0.5, 0]);
xlb = xlabel({'$$k$$';'\vspace{-0.2cm}'}, 'FontSize', labelFontSize, 'FontAngle', 'normal', 'Interpreter', 'latex');
set(xlb, 'Units', 'Normalized', 'Position', [1/2, -0.065, 0]);

lg = legend([p1, p2, p3], 'Initial point 1', 'Initial point 2', 'Initial point 3');
set(lg, 'FontSize', 9);
% set(lg,'Interpreter','latex');
legend('boxoff');

pos = get(lg, 'Position');
set(lg, 'Position', [pos(1)-0.07, pos(2)-0.03, pos(3)-0.08, pos(4)-0.065]);
pos_ = get(lg, 'Position');

print('ND-|supp(xk)|-LASSO.png', '-dpng');
%% xsol and gsol
axesFontSize = 7;
labelFontSize = 10;
legendFontSize = 9;

resolution = 300; % output resolution
output_size = 300 *[10, 8]; % output size

%%%%%% relative error

figure(112), clf;
set(0,'DefaultAxesFontSize', axesFontSize);
set(gcf,'paperunits','centimeters','paperposition',[-0.05 -0.2 output_size/resolution]);
set(gcf,'papersize',output_size/resolution-[0.8 0.6]);

subplot(9,1,[1:4]),

stem(xsol, 'k.', 'linewidth', 0.8, 'markersize', 8);

grid on;
axis([1, n, -2.5,2.5]);
set(gca, 'yTick', -2:1:2);

ylb = ylabel({'$$x^\star$$';'\vspace*{-1.5cm}'},...
    'FontSize', labelFontSize, 'FontAngle', 'normal', 'Interpreter', 'latex');
set(ylb, 'Units', 'Normalized', 'Position', [-0.03, 0.5, 0]);
xlb = xlabel({'$$k$$';'\vspace{-0.2cm}'}, 'FontSize', labelFontSize, 'FontAngle', 'normal', 'Interpreter', 'latex');
set(xlb, 'Units', 'Normalized', 'Position', [1/2, -0.15, 0]);


subplot(9,1,[6:9]),

stem(gsol/mu, 'k.', 'linewidth', 0.8, 'markersize', 8);

grid on;
axis([1, n, -5/4, 5/4]);
set(gca, 'yTick', [-1:2/4:1]);

ylb = ylabel({'$$(\mathcal{K}^T b - x^\star)/\mu$$';},...
    'FontSize', labelFontSize, 'FontAngle', 'normal', 'Interpreter', 'latex');
set(ylb, 'Units', 'Normalized', 'Position', [-0.09, 0.5, 0]);
xlb = xlabel({'$$k$$';'\vspace{-0.2cm}'}, 'FontSize', labelFontSize, 'FontAngle', 'normal', 'Interpreter', 'latex');
set(xlb, 'Units', 'Normalized', 'Position', [1/2, -0.15, 0]);


print('ND-xsol-gsol-LASSO.png', '-dpng');