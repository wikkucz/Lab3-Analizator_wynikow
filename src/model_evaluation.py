from sklearn.metrics import mean_squared_error, r2_score

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

with open('data/evaluation.txt', 'w') as f:
    f.write(f'MSE: {mse}\nR2: {r2}\n')
