import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Wczytaj dane
df = pd.read_csv('processed_data.csv')

# Przygotowanie danych
X = df.drop('score', axis=1)
y = df['score']

# Konwersja zmiennych kategorycznych na dummy
X = pd.get_dummies(X, drop_first=True)

# Zapisz zmodyfikowany zbiór danych do pliku
X['score'] = y  # Dodaj kolumnę 'score' do X, aby móc później ocenić model
X.to_csv('processed_data_with_dummies.csv', index=False)

# Zapisz kolumny, które zostały użyte do nauki
features_used = X.columns.tolist()  # Zapisz wszystkie kolumny
with open('features_used.txt', 'w') as f:
    for feature in features_used:
        f.write(f"{feature}\n")

# Podział danych na zestaw treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Trenowanie modelu
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Zapisz model
joblib.dump(model, 'trained_model.joblib')

print("Model training completed and saved to 'trained_model.joblib'")
