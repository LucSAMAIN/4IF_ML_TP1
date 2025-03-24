import pandas as pd
import os
from sklearn.linear_model import LinearRegression

sample_df = pd.read_csv('insa-ml-2025-regression/sample_submission.csv')
train_df = pd.read_csv('insa-ml-2025-regression/train.csv')
test_df = pd.read_csv('insa-ml-2025-regression/test.csv')

print(train_df.head())
X_train = train_df[[col for col in train_df.columns if col != 'co2' and col != 'id']]
Y_train = train_df['co2']

X_test = test_df[[col for col in test_df.columns if col != 'id']]

reg = LinearRegression().fit(X_train, Y_train)
print(reg.score(X_train, Y_train))

