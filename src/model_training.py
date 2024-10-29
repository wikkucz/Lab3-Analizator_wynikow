from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

X = df.drop('score', axis=1)
y = df['score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestRegressor()
model.fit(X_train, y_train)

joblib.dump(model, 'data/model.joblib')
