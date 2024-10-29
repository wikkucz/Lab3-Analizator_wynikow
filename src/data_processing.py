import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('https://vincentarelbundock.github.io/Rdatasets/csv/AER/CollegeDistance.csv')

numeric_columns = df.select_dtypes(include=['number']).columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())

sns.pairplot(df)
plt.savefig('data/pairplot.png')

print(df.head())
