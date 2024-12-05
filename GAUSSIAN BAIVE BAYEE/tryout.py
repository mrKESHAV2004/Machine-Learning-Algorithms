import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = pd.read_csv('iris.csv')

X = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].apply(pd.to_numeric, errors='coerce')
y = data['label']

data = data.dropna()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

new_data = [[2.3,1.6,1.2,2.1]]  # Example for Iris-virginica

predictions = model.predict(new_data)
print("\nPredictions for New Data:")
print(f"Data: {new_data} -> Predicted Label: {predictions}")
