function [x, its, dk, sk] = func_FB(x0,xsol, GradF, FBS)

maxits = 1e3;
ToL = 1e-14;

x = x0;

dk = zeros(maxits, 1);
sk = zeros(maxits, 1);

its = 1;
while(its<maxits)
        
    g = GradF(x);
    x = FBS(x, g);
        
    %%% stop?
    normE = norm(x(:)-xsol(:), 'fro');
    itsprint(sprintf('      step %07d: norm(ek) = %.3e', its,normE), its);
    
    if ((normE)<ToL)||(normE>1e10); break; end
    
    dk(its) = normE;
    sk(its) = sum(abs(x)>0);
    
    its = its + 1;
    
end
fprintf('\n');

its = its - 1;
dk = dk(1:its);
sk = sk(1:its);