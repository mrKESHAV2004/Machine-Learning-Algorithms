import pandas as pd
import numpy as np

# Create the DataFrame
df = pd.DataFrame({
    'col1': [12, 23, 14, 15],
    'col2': [67, 54, 32, 1],
    'col3': [34, 23, 56, 23]
})

def pearson_correlation(x, y):
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    
    n = np.sum((x - mean_x) * (y - mean_y))
    d = np.sqrt(np.sum((x - mean_x) ** 2) * np.sum((y - mean_y) ** 2))
    
    return n / d

print(pearson_correlation(df['col1'], df['col2']))
print(pearson_correlation(df['col1'], df['col3']))
print(pearson_correlation(df['col2'], df['col3']))

