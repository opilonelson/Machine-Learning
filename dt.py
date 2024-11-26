import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
dfee=pd.read_csv('EEGEyeState.csv')

target=dfee['eyeDetection']
features=dfee.drop('eyeDetection',axis=1)

x_train,x_test,y_train,y_test=train_test_split(features,target,test_size=0.2,random_state=

dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)

y_pred=dt.predict(x_test)
y_pred

print(accuracy_score(y_test,y_pred))
