import pandas as pd 
import numpy as np 
import pickle
import yaml
import os

from sklearn.ensemble import GradientBoostingClassifier

params=yaml.safe_load(open('params.yaml','r'))['model_building']
#fetch the data

train_data=pd.read_csv('./data/features/train_bow.csv')


X_train= train_data.iloc[:,0:-1].values
y_train=train_data.iloc[:,-1].values


clf= GradientBoostingClassifier(n_estimators=params['n_estimators'],learning_rate=params['learning_rate'])

clf.fit(X_train,y_train)

#save the model

pickle.dump(clf,open('model.pkl','wb'))