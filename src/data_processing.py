import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('https://vincentarelbundock.github.io/Rdatasets/csv/AER/CollegeDistance.csv')

df.fillna(df.median(), inplace=True)

sns.pairplot(df)
plt.savefig('data/pairplot.png')
