function [x, its, ek, sk, gamma] = func_SAGA(para, GradF,iGradF, ProxJ, xsol)
%

fprintf(sprintf('performing SAGA...\n'));
itsprint(sprintf('      step %09d: norm(ek) = %.3e', 1,1), 1);

% parameters
n = para.n;
m = para.m;
gamma = para.c_gamma * para.beta_fi;
tau = para.mu * gamma;

% stop cnd, max iteration
ToL = 1e-9;
maxits = 5e6;

% Forward-Backward Step
FB = @(g, tau) ProxJ(g, tau);

% inertial point
x0 = zeros(n, 1);

G = zeros(n, m);
for i=1:m
    G(:, i) = iGradF(x0, i);
end

%
%%%%%%%%%%%%%%%%%%%%%%%%%%
%

%%% obtain the minimizer x^\star
ek = zeros(1, maxits);
sk = zeros(1, maxits);

x = x0; % xk
x_k1 = x;

its = 1;
while(its<maxits)
    
    x_k2 = x_k1;
    x_k1 = x;
    
    
    j = mod(its-1, m) + 1;
    % j = randperm(m, 1);
    
    gj_old = G(:, j);
    
    gj = iGradF(x_k1, j);
    
    w = x - gamma* (gj - gj_old) - gamma/m* sum(G, 2);
    x = wthresh(w, 's', tau);
    
    x(end) = w(end);
    
    G(:, j) = gj;
    
    %%% stop?
    normE = norm(x(:)-xsol(:), 'fro');
    if mod(its,1e3)==0; itsprint(sprintf('      step %09d: norm(ek) = %.3e', its,normE), its); end
    
    ek(its) = normE;
    if ((normE)<ToL)||(normE>1e10); break; end
    
    sk(its) = sum(abs(x) > 0);
    
    its = its + 1;
    
end
fprintf('\n');

ek = ek(1:its-1);
sk = sk(1:its-1);
