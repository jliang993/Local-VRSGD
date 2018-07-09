function [x, its, ek, fk, sk, gk, vk] = func_acc_SAGA(para, iGradF, ObjF)
%

fprintf(sprintf('performing acc-SAGA...\n'));
itsprint(sprintf('      step %09d: norm(ek) = %.3e', 1,1), 1);

% parameters
n = para.n;
m = para.m;
gamma0 = para.c_gamma * para.beta_fi;
gamma = gamma0;
tau = para.mu * gamma;

% stop cnd, max iteration
tol = para.tol;
maxits = para.maxits;

W = para.W;

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
ek = zeros(maxits, 1);
sk = zeros(maxits, 1);
gk = zeros(maxits, 1);
fk = zeros(maxits, 1);
vk = zeros(maxits, 1);

g_flag = 0;

x = x0; % xk

l = 0;

its = 1;
while(its<maxits)
    
    x_old = x;
    
    % j = mod(its-1, m) + 1;
    j = randperm(m, 1);
    
    gj_old = G(:, j);
    
    gj = iGradF(x_old, j);
    
    w = x - gamma* (gj - gj_old) - gamma/m* sum(G, 2);
    x = wthresh(w, 's', tau);
    
    x(end) = w(end);
    
    G(:, j) = gj;
    
    %%% stop?
    normE = norm(x(:)-x_old(:), 'fro');
    if mod(its,1e3)==0; itsprint(sprintf('      step %09d: norm(ek) = %.3e', its,normE), its); end

    sk(its) = sum(abs(x) > 0);
    gk(its) = gamma;
    
    if (its>m)&&(mod(its, m)==0)&&(sk(its)<n/2)&&(var(sk(its-m+1:its))<1)
        PT = diag(double(abs(x)>0));
        WT = W*PT;
        b = zeros(m, 1);
        for i=1:m
            WTi = WT(i,:);
            b(i) = norm(WTi)^2 /4;
        end
        beta_fi_new = 1 /max(b);
        
        E = beta_fi_new /para.beta_fi;
        g_flag = 1;
    end

    if (its>50*m)
        vk(its) = var(sk(its-m+1:its));
    end
    
    if (g_flag)&&(mod(its, m)==0)
        
        gamma = min(gamma*1.5, E*gamma0);
        tau = para.mu * gamma;
        
    end
    
    ek(its) = normE;
    if ((normE)<tol)||(normE>1e10); break; end
    
    if mod(its, m)==0
        l = l + 1;
        fk(l) = ObjF(x);
    end
    
    its = its + 1;
    
end
fprintf('\n');

% g_flag

ek = ek(1:its-1);
sk = sk(1:its-1);
gk = gk(1:its-1);
vk = vk(1:its-1);
fk = fk(1:l);
