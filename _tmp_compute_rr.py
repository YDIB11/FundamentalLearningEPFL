import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, ShuffleSplit, GridSearchCV
from sklearn import preprocessing, linear_model
from sklearn.pipeline import make_pipeline

data = pd.read_csv('https://web.stanford.edu/~hastie/ElemStatLearn/datasets/LAozone.data')
y = data['ozone'].to_numpy().astype(float)
X = data[data.columns[1:]].to_numpy()

XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.25, random_state=1)

regRR = linear_model.Ridge(alpha=0.01)
pipe_regRR = make_pipeline(preprocessing.StandardScaler(), regRR)

alphas = np.logspace(-5, 3, 200)
cv = ShuffleSplit(n_splits=30, test_size=0.05, random_state=1)

def neg_rss(estimator, X, y):
    pred = estimator.predict(X)
    return -np.mean((y - pred) ** 2)

cv_regRR = GridSearchCV(
    pipe_regRR,
    param_grid={'ridge__alpha': alphas},
    cv=cv,
    scoring=neg_rss,
    return_train_score=True,
)

cv_regRR.fit(XTrain, yTrain)
best_alpha = cv_regRR.cv_results_['param_ridge__alpha'][cv_regRR.best_index_]

best_pipe = cv_regRR.best_estimator_
best_pipe.fit(XTrain, yTrain)

train_pred = best_pipe.predict(XTrain)
test_pred = best_pipe.predict(XTest)
train_rss = np.mean((yTrain - train_pred) ** 2)
test_rss = np.mean((yTest - test_pred) ** 2)

scaler = preprocessing.StandardScaler().fit(XTrain)
XTrain_scaled = scaler.transform(XTrain)
XTest_scaled = scaler.transform(XTest)
yTrain_mean = yTrain.mean()
yTrain_centered = yTrain - yTrain_mean
regL = linear_model.LinearRegression().fit(XTrain_scaled, yTrain_centered)
ols_train_pred = regL.predict(XTrain_scaled) + yTrain_mean
ols_test_pred = regL.predict(XTest_scaled) + yTrain_mean
ols_train_rss = np.mean((yTrain - ols_train_pred) ** 2)
ols_test_rss = np.mean((yTest - ols_test_pred) ** 2)

print('best_alpha', best_alpha)
print('ridge train RSS', train_rss)
print('ridge test RSS', test_rss)
print('OLS train RSS', ols_train_rss)
print('OLS test RSS', ols_test_rss)
print('ridge coef', best_pipe.named_steps['ridge'].coef_)
print('ridge intercept', best_pipe.named_steps['ridge'].intercept_)
