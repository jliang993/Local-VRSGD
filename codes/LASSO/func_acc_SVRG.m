function [x, t, ek, fk, sk, gk] = func_acc_SVRG(para, GradF,iGradF, ObjF)
%

fprintf( sprintf('performing SVRG...\n') );
itsprint(sprintf('      step %09d: norm(ek) = %.3e...', 1,1), 1);


% parameters
P = para.P;
m = para.m;
n = para.n;
W = para.W;
gamma = para.c_gamma * para.beta_fi;
tau = para.mu * gamma;

% stop cnd, max iteration
tol = para.tol;
% maxits = para.maxits;
maxits = 5e4;


% initial point
x0 = zeros(n, 1);

%
%%%%%%%%%%%%%%%%%%%%%%%%%%
%

ek = zeros(maxits, 1);
fk = zeros(maxits, 1);
sk = zeros(maxits, 1);
gk = zeros(maxits, 1);

x = x0;
x_tilde = x;

l = 0;

its = 1;
t = 1;
while(t<maxits)
    
    g_tilde = GradF(x_tilde);
    
    x = x_tilde;
    for p=1:P
        
        x_old = x;
        
        j = randsample(1:m,1);
        
        Gj_k1 = iGradF(x, j);
        Gj_k2 = iGradF(x_tilde, j);
        
        w = x - gamma* ( Gj_k1 - Gj_k2 + g_tilde );
        x = wthresh(w, 's', tau);
        
        %%% stop?
        normE = norm(x(:)-x_old(:), 'fro');
        
        if mod(t, m)==0
            l = l + 1;
            fk(l) = ObjF(x);
        end
        if mod(t,1e3)==0; itsprint(sprintf('      step %09d: norm(ek) = %.3e...', t,normE), t); end
        
        ek(t) = normE;
        sk(t) = sum(abs(x)>0);
        gk(t) = gamma;
        
        if (mod(t, m)==0)&&(var(sk(t-m+1:t))<1e-1)
            PT = diag(double(abs(x)>0));
            WT = W*PT;
            b = zeros(m, 1);
            for i=1:m
                WTi = WT(i,:);
                b(i) = norm(WTi)^2;
            end
            beta_fi = 1 /max(b);
            
            gamma = max(para.c_gamma * beta_fi /10, gamma);
            tau = para.mu * gamma;
        end
        
        %%%% the change of gamma should be smoothly!!!!!
        
        t = t + 1;
        
    end
    
    x_tilde = x;
    
    %%% stop?
    if ((normE)<tol)||(normE>1e10)||(isnan(normE)); break; end
    
    its = its + 1;
    
end
fprintf('\n');

ek = ek(1:t-1);
fk = fk(1:l);
sk = sk(1:t-1);
gk = gk(1:t-1);


% function [x, its,t, dl,dk, sk, gamma] = func_acc_SVRG(para, GradF,iGradF, ProxJ, xsol)
% %
% 
% fprintf(sprintf('performing acc SVRG...\n'));
% itsprint(sprintf('      step %09d: norm(ek) = %.3e', 1,1), 1);
% 
% 
% % parameters
% P = para.P;
% m = para.m;
% n = para.n;
% gamma = para.c_gamma * para.beta_fi;
% tau = para.mu * gamma;
% 
% W = para.W;
% 
% % stop cnd, max iteration
% ToL = 1e-10;
% maxits = 3e6;
% 
% 
% % initial point
% x0 = zeros(n, 1);
% 
% %
% %%%%%%%%%%%%%%%%%%%%%%%%%%
% %
% 
% dl = zeros(1, maxits);
% dk = zeros(1, maxits);
% sk = zeros(1, maxits);
% 
% 
% x = x0;
% x_tilde = x;
% 
% its = 1;
% t = 1;
% while(its<maxits)
%     
%     mu = GradF(x_tilde);
%     
%     x = x_tilde;
%     for p=1:P
%         
%         j = randsample(1:m,1);
%         
%         Gj_k1 = iGradF(x, j);
%         Gj_k2 = iGradF(x_tilde, j);
%         
%         w = x - gamma* ( Gj_k1 - Gj_k2 + mu );
%         x = wthresh(w, 's', tau);
%         
%         x(end) = w(end);
%         
%         distE = norm(x(:)-xsol(:), 'fro');
%         if mod(t,5e4)==0; itsprint(sprintf('      step %09d: norm(ek) = %.3e', t,distE), t); end
%         
%         sk(t) = sum(abs(x) > 0);
%         dk(t) = distE;
%         
%         t = t + 1;
%         
%         if mod(t, 5e3)==0
%             PT = diag(double(abs(x)>0));
%             WT = W*PT;
%             b = zeros(m, 1);
%             for i=1:m
%                 WTi = WT(i,:);
%                 b(i) = norm(WTi)^2 /4;
%             end
%             beta_fi = 1 /max(b);
%             
%             gamma = para.c_gamma * beta_fi;
%             tau = para.mu * gamma;
%         end
%         
%     end
%     
%     x_tilde = x;
%     
%     %%% stop?
%     normE = norm(x_tilde(:)-xsol(:), 'fro');
%     % if mod(its,1e2)==0; itsprint(sprintf('      step %08d: norm(ek) = %.3e', its,normE), its); end
%     
%     dl(its) = normE;
%     if ((normE)<ToL)||(normE>1e10); break; end
%     
%     its = its + 1;
%     
% end
% fprintf('\n');
% 
% dl = dl(1:its-1);
% sk = sk(1:t-1);
% dk = dk(1:t-1);
