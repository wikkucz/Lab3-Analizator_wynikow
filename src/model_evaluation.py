import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

df = pd.read_csv('processed_data.csv')

X = df.drop('score', axis=1)
y = df['score']

X = pd.get_dummies(X, drop_first=True)

model = joblib.load('trained_model.joblib')

_, X_test, _, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Model evaluation completed:\nMSE: {mse}\nMAE: {mae}\nR2: {r2}")
