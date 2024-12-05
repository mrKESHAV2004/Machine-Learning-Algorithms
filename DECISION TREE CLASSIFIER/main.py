import numpy as np 
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn import tree

dataset = pd.read_csv("winequalityN.csv")

print(dataset.head())
print("Shape of the dataset:", dataset.shape)

lb = LabelEncoder()
dataset['type'] = lb.fit_transform(dataset['type'])

x = dataset.drop("quality", axis=1)
y = dataset["quality"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

dtree = DecisionTreeClassifier(criterion="entropy", max_depth=10)
dtree = dtree.fit(x_train, y_train)

prediction = dtree.predict(x_test)

acc = metrics.accuracy_score(prediction, y_test)
print("Accuracy of the model:", acc)

cnf_matrix = metrics.confusion_matrix(y_test, prediction)
print("Confusion Matrix:\n", cnf_matrix)

plt.figure(figsize=(20, 10))
tree.plot_tree(dtree, feature_names=list(x.columns), class_names=[str(i) for i in sorted(y.unique())], filled=True)
plt.show()