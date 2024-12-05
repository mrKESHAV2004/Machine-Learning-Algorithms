import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix

# Load the dataset
dataset = pd.read_csv('iris.csv')

x = dataset.drop('label', axis='columns')
y = dataset['label']

train_data, test_data, train_label, test_label = train_test_split(x, y, test_size=0.3)

model = KNeighborsClassifier(n_neighbors=5)

model.fit(train_data, train_label)

prediction = model.predict(test_data)

acc = metrics.accuracy_score(prediction, test_label)
print("Accuracy:", acc)

x_new = ['sepal_length','sepal_width','petal_length','petal_width']
data = [5.1,3,1,2]
patient = pd.DataFrame([data], columns=x_new)

predict_defect = model.predict(patient)
print(predict_defect)

conf_matrix = confusion_matrix(test_label, prediction)
print("Confusion Matrix:")
print(conf_matrix)
