import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, linear_model
from sklearn.pipeline import make_pipeline

data = pd.read_csv('https://web.stanford.edu/~hastie/ElemStatLearn/datasets/LAozone.data')
y = data['ozone'].to_numpy().astype(float)
X = data[data.columns[1:]].to_numpy()
XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.25, random_state=1)

alpha = 11.49756995397738
pipe = make_pipeline(preprocessing.StandardScaler(), linear_model.Ridge(alpha=alpha))
pipe.fit(XTrain, yTrain)
train_pred = pipe.predict(XTrain)
test_pred = pipe.predict(XTest)
train_rss = np.mean((yTrain - train_pred)**2)
test_rss = np.mean((yTest - test_pred)**2)

scaler = preprocessing.StandardScaler().fit(XTrain)
XTrain_scaled = scaler.transform(XTrain)
XTest_scaled = scaler.transform(XTest)
yTrain_mean = yTrain.mean()
yTrain_centered = yTrain - yTrain_mean
regL = linear_model.LinearRegression().fit(XTrain_scaled, yTrain_centered)
ols_train_pred = regL.predict(XTrain_scaled) + yTrain_mean
ols_test_pred = regL.predict(XTest_scaled) + yTrain_mean
ols_train_rss = np.mean((yTrain - ols_train_pred)**2)
ols_test_rss = np.mean((yTest - ols_test_pred)**2)

print('ridge train rss', train_rss)
print('ridge test rss', test_rss)
print('ols train rss', ols_train_rss)
print('ols test rss', ols_test_rss)
print('coeff', pipe.named_steps['ridge'].coef_)
