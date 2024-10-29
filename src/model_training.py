import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

df = pd.read_csv('processed_data.csv')

X = df.drop('score', axis=1)
y = df['score']

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, 'trained_model.joblib')

print("Model training completed and saved to 'trained_model.joblib'")

