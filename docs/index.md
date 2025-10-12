# Teammate theorist

We have implemented two separate theorist as team-mate:
 - For the hackathon we have implemented a deep learning based theorist. It makes use of differentiable, but discrete (via differentiable gates, implemented from hard concrete distribution) function tree optimization.
   It uses lagrangian multipliers for the constrained optimization. However this approach is hard to optimize for the grading system employed by the course
 - For this reason we have implemented a second theorist, ParallelizedAkaikeBMSRegressorPool which we are submitting for the coursework. This theorist implements a pool of Bayesian Machine Scientists (BMS) as 
   described in Guimerà et al., 2020. While BMS is highly efficient, it runs on a single core. Additionally, repeated BMS runs can result in hugely different models, with strongly varying performance.
   To mitigate this, our ParallelizedAkaikeBMSRegressorPool first fits a set of BMS models in parallel (the pool arg can e.g. be set to the number of CPU cores). 
   Then, it selects a subset of the best performing models based on a naive estimate of the Akaike Information Criterion (AIC), selecting better performing models with shorter equations.
   

We evaluate the performance of the ParallelizedAkaikeBMSRegressorPool on the three benchmark tasks made available in the course, see akaike_bms.py / main:
 - Benchmarking Steven's Power Law:
    - Validation MSE: 7.307167266447019e-07
    - Winning Equation: (X0 ** 0.8)
 - Benchmarking Weber-Fechner Law
    - Validation MSE: 9.930924676040441e-07
    - Winning Equation: log((X1 / X0))
 - Benchmarking Expected Value Theory
   - Validation MSE: 0.0009514312515582806
   - Winning Equation: sin((6.44 ** abs(X3)))

The search space for the BMS is simply the default operator space, equation depth is both limited by the BMS algorithm design, as well as the AIC selection.

We test the ParallelizedAkaikeBMSRegressorPool with two separate test cases in test_akaike_bms.py

