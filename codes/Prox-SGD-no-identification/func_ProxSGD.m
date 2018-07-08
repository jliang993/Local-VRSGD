function [x, its, ek, sk] = func_ProxSGD(x0,y,A, gamma, ProxSGD)
itsprint(sprintf('      step %09d: norm(ek) = %.3e', 1,1), 1);

maxits = 1e6 + 1;
ToL = 1e-14;

ek = zeros(1, maxits);
sk = zeros(1, maxits);

p = 0.51;

x = x0;

its = 1;
while(its<maxits)
    
    x_old = x;
    
    j = randperm(3, 1);
    Aj = A(j, :);
    
    gj = (Aj*x_old - y(j)) *(Aj');
    
    gammak = gamma/ its^(p);
    x = ProxSGD(x, gj, gammak);
    
    %%% stop?
    normE = norm(x(:)-x_old(:), 'fro');
    if mod(its, 1e4)==0; itsprint(sprintf('      step %09d: norm(ek) = %.3e', its,normE), its); end

    if ((normE)<ToL)||(normE>1e10); break; end
    
    ek(its) = normE;
    sk(its) = sum(abs(x)>1e-15);
    
    its = its + 1;
    
end
fprintf('\n');

its = its - 1;
ek = ek(1:its);
sk = sk(1:its);