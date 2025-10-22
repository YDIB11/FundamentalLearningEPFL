import json
path="TP6/regularized_linear_regression.ipynb"
with open(path,'r',encoding='utf-8') as f:
    nb=json.load(f)

nb['cells'][68]['source']=["**Q6. Behaviour at the extremes of $\\alpha$**\n", "- $\\alpha=0$: the penalty disappears, so Ridge and Lasso both coincide with the ordinary least-squares estimator (with our standardised data the coefficients exactly match OLS).\n", "- $\\alpha\\to\\infty$: the penalty dominates; Ridge shrinks all weights to zero, Lasso forces every coefficient exactly to zero, leaving only the intercept. Consequently both predict the constant average response.\n"]

nb['cells'][69]['source']=["**Q6. Qualitative comparison of regularisation paths**\n", "- The figure above shows the coefficient paths against $\\alpha$ on a linear axis. Ridge curves decay smoothly and never touch zero; every feature keeps a small weight even for large $\\alpha$.\n", "- Lasso paths contain sharp kinks: features drop to exactly zero at specific thresholds and remain flat thereafter, illustrating the sparsity-inducing nature of the $\\ell_1$ penalty.\n"]

with open(path,'w',encoding='utf-8') as f:
    json.dump(nb,f,indent=1)
