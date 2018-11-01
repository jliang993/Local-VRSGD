# Local Convergence Behaviour of SAGA/ProxSVRG

Matlab code to reproduce the results of the paper

[Local Convergence Properties of SAGA/Prox-SVRG and Beyond](http://proceedings.mlr.press/v80/poon18a.html)

[Clarice Poon](http://www.damtp.cam.ac.uk/user/cmhsp2/), [Jingwei Liang](https://jliang993.github.io/), [Carola-Bibiane Schönlieb](http://www.damtp.cam.ac.uk/user/cbs31/Home.html), 2018



## Prox-SGD has no manifold identification

![Prox-SGD has no identification](codes/Prox-SGD-no-identification/Supp-ProxSGD-LASSO.png)



## When non-degeneracy condition fails

Solution and its dual          |  Support identification of three different initial points
:-------------------------:|:-------------------------:
![ ](codes/ND-fails/ND-xsol-gsol-LASSO.png)  |  ![ ](codes/ND-fails/ND-supp(xk)-LASSO.png)



## Sparse Logistic Regression

### Toy example

Support identification of SAGA/Prox-SVRG          |  Local linear convergence of SAGA/Prox-SVRG
:-------------------------:|:-------------------------:
![ ](codes/Sparse-LogReg/toy_sagasvrg_slr_sk.png)  |  ![ ](codes/Sparse-LogReg/toy_sagasvrg_slr_objf.png)



## LASSO

Support identification of SAGA/Prox-SVRG          |  Local linear convergence of SAGA/Prox-SVRG
:-------------------------:|:-------------------------:
![ ](codes/LASSO/sagasvrg_lasso_gisette_sk.png)  |  ![ ](codes/LASSO/sagasvrg_lasso_gisette_objf.png)

Copyright (c) 2018 Clarice Poon and  Jingwei Liang