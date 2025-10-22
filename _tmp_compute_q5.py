import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import ShuffleSplit, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing, linear_model

# ensure function consistent with notebook
import pandas as pd

diabetes = load_diabetes()
X_diabetes = diabetes.data
y_diabetes = diabetes.target

rng = np.random.RandomState(0)
indices = rng.permutation(len(X_diabetes))
train_idx = indices[:150]
test_idx = indices[150:200]
XTrain_diab, XTest_diab = X_diabetes[train_idx], X_diabetes[test_idx]
yTrain_diab, yTest_diab = y_diabetes[train_idx], y_diabetes[test_idx]

alphas = np.logspace(-2, 5, 100)
cv = ShuffleSplit(n_splits=40, test_size=0.2, random_state=0)

def neg_rss(estimator, X, y):
    preds = estimator.predict(X)
    return -np.mean((y - preds)**2)

pipe_regRR = make_pipeline(preprocessing.StandardScaler(), linear_model.Ridge())
pipe_regLasso = make_pipeline(preprocessing.StandardScaler(), linear_model.Lasso(max_iter=10000))

cv_rr_diab = GridSearchCV(pipe_regRR, [{'ridge__alpha': alphas}], cv=cv, scoring=neg_rss, return_train_score=True)
cv_lasso_diab = GridSearchCV(pipe_regLasso, [{'lasso__alpha': alphas}], cv=cv, scoring=neg_rss, return_train_score=True)

cv_rr_diab.fit(XTrain_diab, yTrain_diab)
cv_lasso_diab.fit(XTrain_diab, yTrain_diab)

best_alpha_rr = float(cv_rr_diab.best_params_['ridge__alpha'])
best_alpha_lasso = float(cv_lasso_diab.best_params_['lasso__alpha'])

print(best_alpha_rr, best_alpha_lasso)

# compute best models RSS on hold-out test
pipe_regRR.set_params(ridge__alpha=best_alpha_rr)
pipe_regRR.fit(XTrain_diab, yTrain_diab)
pipe_regLasso.set_params(lasso__alpha=best_alpha_lasso)
pipe_regLasso.fit(XTrain_diab, yTrain_diab)

ridge_test_rss = np.mean((yTest_diab - pipe_regRR.predict(XTest_diab))**2)
lasso_test_rss = np.mean((yTest_diab - pipe_regLasso.predict(XTest_diab))**2)
print('ridge test RSS', ridge_test_rss)
print('lasso test RSS', lasso_test_rss)

# gather curves (for discussion maybe not necessary)

# store results in file for update? Instead print values.
