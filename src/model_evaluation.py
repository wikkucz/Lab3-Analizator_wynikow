import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score

# Wczytaj kolumny użyte do nauki
with open('features_used.txt', 'r') as f:
    features_used = [line.strip() for line in f]

# Wczytaj dane
df = pd.read_csv("processed_data_with_dummies.csv")

# Sprawdź, jakie kolumny są w DataFrame
print("Dostępne kolumny w df:", df.columns.tolist())

# Oddziel cechy (X) od zmiennej docelowej (y)
X = df[features_used]  # Użyj kolumn, które zostały zapisane
y = df['score']

# Wczytaj model
model = joblib.load('trained_model.joblib')

# Predykcja
y_pred = model.predict(X)

# Ocena modelu
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f'MSE: {mse}')
print(f'R²: {r2}')
