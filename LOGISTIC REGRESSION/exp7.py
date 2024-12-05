import pandas as pd
import numpy as np 
dataset = pd.read_csv('diabetes2.csv')
dataset.head()
x = dataset.drop('Outcome',axis='columns')
y = dataset['Outcome']
print(x)
print(y)
from sklearn.model_selection import train_test_split
train_data,test_data,train_label,test_label = train_test_split(x,y,test_size=0.3)
print(train_data.shape)
print(test_data.shape)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(train_data,train_label)
from sklearn import metrics
prediction = model.predict(test_data)
acc = metrics.accuracy_score(prediction,test_label)
print(acc)
x= ['Glucose','BloodPressure','BloodGroup','Infection','Weight','Age']
data = [138,52,'A',43.6,32,34]
patient = pd.DataFrame([data],columns=x)
patient.head()
predict_diabetes = model.predict(patient)
print(predict_diabetes)