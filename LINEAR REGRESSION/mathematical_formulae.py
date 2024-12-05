import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def linear_regression(X, y):
    n = len(X)
    sum_x = np.sum(X)
    sum_y = np.sum(y)
    sum_xy = np.sum(X*y)
    sum_x_square = np.sum(X**2)

    m = (n * sum_xy - sum_x * sum_y) / (n * sum_x_square - sum_x**2)
    b = (sum_y - m * sum_x) / n
    predicted_y = m * X + b
    plt.scatter(X, y, label='Actual Salary')
    plt.plot(X, predicted_y, color='red', label='Predicted Salary')
    plt.xlabel('Years of Experience')
    plt.ylabel('Salary')
    plt.title('Salary vs Years of Experience (Linear Regression)')
    plt.legend()
    plt.grid(True)
    plt.show()
    return m, b

data = pd.read_csv("Salary.csv")
df = pd.DataFrame(data)

X = data['YearsExperience'].values
y = data['Salary'].values

m, b = linear_regression(X, y)
new_experience = float(input("Enter years of experience: "))
predicted_salary = m * new_experience + b
print("Predicted salary:", predicted_salary)
