function [x, its,t, dl,dk, sk, gamma] = func_SVRG(para, GradF,iGradF, ProxJ, xsol)
%

fprintf(sprintf('performing SVRG...\n'));
itsprint(sprintf('      step %09d: norm(ek) = %.3e', 1,1), 1);


% parameters
P = para.P;
m = para.m;
n = para.n;
gamma = para.c_gamma * para.beta_fi;
tau = para.mu * gamma;

% stop cnd, max iteration
ToL = 1e-9;
maxits = 3e6;


% initial point
x0 = zeros(n, 1);

%
%%%%%%%%%%%%%%%%%%%%%%%%%%
%

dl = zeros(1, maxits);
dk = zeros(1, maxits);
sk = zeros(1, maxits);


x = x0;
x_tilde = x;

its = 1;
t = 1;
while(its<maxits)
    
    mu = GradF(x_tilde);
    
    x = x_tilde;
    for p=1:P
        
        j = randsample(1:m,1);
        
        Gj_k1 = iGradF(x, j);
        Gj_k2 = iGradF(x_tilde, j);
        
        w = x - gamma* ( Gj_k1 - Gj_k2 + mu );
        x = wthresh(w, 's', tau);
        
        x(end) = w(end);
        
        distE = norm(x(:)-xsol(:), 'fro');
        if mod(t,5e4)==0; itsprint(sprintf('      step %09d: norm(ek) = %.3e', t,distE), t); end
        
        sk(t) = sum(abs(x) > 0);
        dk(t) = distE;
        
        t = t + 1;
        
    end
    
    x_tilde = x;
    
    %%% stop?
    normE = norm(x_tilde(:)-xsol(:), 'fro');
    % if mod(its,1e2)==0; itsprint(sprintf('      step %08d: norm(ek) = %.3e', its,normE), its); end
    
    dl(its) = normE;
    if ((normE)<ToL)||(normE>1e10); break; end
    
    its = its + 1;
    
end
fprintf('\n');

dl = dl(1:its-1);
sk = sk(1:t-1);
dk = dk(1:t-1);
