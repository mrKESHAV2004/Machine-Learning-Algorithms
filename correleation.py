import pandas as pd

df = pd.DataFrame({
    'col1': [12, 23, 14, 15],
    'col2': [67, 54, 32, 1],
    'col3': [34, 23, 56, 23]
})

print(df)
print(df['col1'].corr(df['col2']))
print(df['col2'].corr(df['col3']))
print(df['col1'].corr(df['col3']))