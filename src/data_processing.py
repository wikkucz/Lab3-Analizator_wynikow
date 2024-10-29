import pandas as pd

df = pd.read_csv('https://vincentarelbundock.github.io/Rdatasets/csv/AER/CollegeDistance.csv')

numeric_columns = df.select_dtypes(include=['number']).columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())

df.to_csv('processed_data.csv', index=False)

print("Data processing completed and saved to 'processed_data.csv'")

