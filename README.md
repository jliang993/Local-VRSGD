# Local Convergence Behaviour of SAGA/ProxSVRG

Matlab code to reproduce the results of the paper

Local Convergence Properties of SAGA/Prox-SVRG and Beyond

Clarice Poon, Jingwei Liang, Carola-Bibiane Schönlieb



## Prox-SGD has no manifold identification

![Prox-SGD has no identification](codes/Prox-SGD-no-identification/Supp-ProxSGD-LASSO.png)



## When non-degeneracy condition fails

Solution and its dual          |  Support identification of three different initial points
:-------------------------:|:-------------------------:
![ ](codes/ND-fails/ND-xsol-gsol-LASSO.png)  |  ![ ](codes/ND-fails/ND-supp(xk)-LASSO.png)



## Sparse Logistic Regression

Support identification of SAGA/Prox-SVRG          |  Local linear convergence of SAGA/Prox-SVRG
:-------------------------:|:-------------------------:
![ ](codes/Sparse-LogReg/LogisticRegression_saga_svrg_support.png)  |  ![ ](codes/Sparse-LogReg/LogisticRegression_saga_svrg_rate.png)




Copyright (c) 2018 Clarice Poon and  Jingwei Liang