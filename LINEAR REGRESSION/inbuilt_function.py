import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv("Salary.csv")
X = data['YearsExperience'].values
y = data['Salary'].values

slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)

_x = float(input("Enter years of experience: "))
predicted_y = slope * _x + intercept
print("Predicted Salary for", _x, "years of experience:", predicted_y)

plt.scatter(X, y, label='Actual Salary', color='blue')
plt.plot(X, slope * X + intercept, color='red', label='Regression Line')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Salary vs Years of Experience (Linear Regression)')
plt.legend()
plt.grid(True)

plt.show()
