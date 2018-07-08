function [x, its, ek, sk] = func_FB(x0, GradF, FBS)

maxits = 1e3;
ToL = 1e-14;

x = x0;

ek = zeros(1, maxits);
sk = zeros(1, maxits);

its = 1;
while(its<maxits)
    
    x_old = x;
    
    g = GradF(x);
    x = FBS(x, g);
    
    %%% stop?
    normE = norm(x(:)-x_old(:), 'fro');
    itsprint(sprintf('      step %09d: norm(ek) = %.3e', its,normE), its);
    
    if ((normE)<ToL)||(normE>1e10); break; end
    
    ek(its) = normE;
    sk(its) = sum(abs(x)>0);
    
    its = its + 1;
    
end
fprintf('\n');

its = its - 1;
ek = ek(1:its);
sk = sk(1:its);