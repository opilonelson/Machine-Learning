import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

df=pd.read_csv('demodata.csv')
df.rename(columns={'feature_1':'temp1','feature_2':'temp2'},inplace=True)
df.isnull().sum().sum()
'''import modules - IsolationForest, StandardScaler
create an instance of the scaler 
scaling
fit_transform
fit_predict
visualize'''
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)
scaled_data

clf = IsolationForest(contamination=0.01, random_state=42)
df['Anomaly'] = clf.fit_predict(scaled_data)
df
df['Anomaly_Descr'] = df['Anomaly'].map({1:'Normal',-1:'Anomaly'})
df
rom matplotlib import pyplot as plt
plt.figure(figsize=(10,6))
plt.plot(df.index,df['temp1'],label='temp1', color='red')
plt.plot(df.index,df['temp2'],label='temp2')


#anomalies = df[df['Anomaly']==-1]

anomalies = df[df['Anomaly_Descr']=='Anomaly']
plt.scatter(anomalies.index, anomalies['temp1'], color='red', label='Anomaly')
plt.scatter(anomalies.index, anomalies['temp2'], color='blue', label='Anomaly')
#plt.legend()
plt.show()
