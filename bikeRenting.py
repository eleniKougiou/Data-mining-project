import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingRegressor

#train
df_train = pd.read_csv('train.csv' )
df_train.rename(columns={'weathersit':'weather','mnth':'month','hr':'hour','yr':'year','hum': 'humidity','cnt':'count'},inplace=True)
df_train['season'] = df_train.season.astype('category')
df_train['year'] = df_train.year.astype('category')
df_train['month'] = df_train.month.astype('category')
df_train['hour'] = df_train.hour.astype('category')
df_train['holiday'] = df_train.holiday.astype('category')
df_train['weekday'] = df_train.weekday.astype('category')
df_train['workingday'] = df_train.workingday.astype('category')
df_train['weather'] = df_train.weather.astype('category')

#test
df_test = pd.read_csv('test.csv')
df_test.rename(columns={'weathersit':'weather','mnth':'month','hr':'hour','yr':'year','hum': 'humidity','cnt':'count'},inplace=True)
df_test['season'] = df_test.season.astype('category')
df_test['year'] = df_test.year.astype('category')
df_test['month'] = df_test.month.astype('category')
df_test['hour'] = df_test.hour.astype('category')
df_test['holiday'] = df_test.holiday.astype('category')
df_test['weekday'] = df_test.weekday.astype('category')
df_test['workingday'] = df_test.workingday.astype('category')
df_test['weather'] = df_test.weather.astype('category')

#OHE
df_train= pd.get_dummies( df_train,columns=['weekday','season','year','month','hour','holiday','workingday','weather'])
df_test = pd.get_dummies( df_test,columns=['weekday', 'season','year','month','hour','holiday','workingday','weather'])


y = df_train['count']
df_train = df_train.drop(['atemp'], axis=1)

df_train, df_test = df_train.align(df_test, join='inner', axis=1)

X = df_train


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

mlpr = MLPRegressor(solver='lbfgs',
                   activation='relu',
                   alpha=0.001,
                   hidden_layer_sizes=(50,50,),
                   random_state=1,
                   max_iter=550,
                   learning_rate_init=0.001)


mlpc= MLPClassifier(solver='lbfgs',
                   activation='relu',
                   alpha=0.00003,
                   hidden_layer_sizes=(50,50,50,50,),
                   random_state=21,
                   max_iter=550,
                   learning_rate_init=0.001)

# Voting
voting = VotingRegressor(estimators=[('mlpr', mlpr), ('mlpc', mlpc)], weights=None, n_jobs=-1)
voting.fit(X_train, y_train)
y_pred = voting.predict(X_test)

True in (y_pred < 0)
for i, y  in enumerate(y_pred):
    if y_pred[i] < 0:
        y_pred[i] = 0
        

print('RMSLE:', np.sqrt(mean_squared_log_error(y_test, y_pred)))
    
y_pred = voting.predict(df_test)

True in (y_pred < 0)
for i, y  in enumerate(y_pred):
    if y_pred[i] < 0:
        y_pred[i] = 0
        


submission = pd.DataFrame()
submission['Id'] = range(y_pred.shape[0])
submission['Predicted'] = y_pred

submission.to_csv("submission.csv", index=False)