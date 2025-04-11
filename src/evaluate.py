import pandas as pd
from sklearn.metrics import r2_score,mean_squared_error,root_mean_squared_error
import pickle

X_test = pd.read_csv('data/X_test.csv')
y_test = pd.read_csv('data/y_test.csv')

with open('models/trained_model.pkl', 'rb') as f:
    grade_prediction_model = pickle.load(f)

# predicting and evaluating the model performance
y_pred = grade_prediction_model.predict(X_test)
r2 = r2_score(y_test,y_pred)
mse = mean_squared_error(y_test,y_pred)
rmse = root_mean_squared_error(y_test,y_pred)
# Adjusted r2 score
n = len(y_test)
k = X_test.shape[1]
adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - k - 1))

# Showing results
print(f"R² score: {r2:.3f}")
print(f"Adjusted R² score: {adjusted_r2:.3f}")
print(f"MSE: {mse:.1f}")
print(f"RMSE: {rmse:.1f}")
print(f"About {(adjusted_r2 * 100):.1f}% variance explained!")  

# Contribution of Study hour and Attendance
feature_names = X_test.columns
coefficients = grade_prediction_model.coef_ 
total_weight = sum(abs(coef) for coef in coefficients)

print("\nHow Much Each Feature Drives Predictions:")
for name, coef in zip(feature_names, coefficients):
    percent = (abs(coef) / total_weight) * 100
    print(f"{name} drove {percent:.2f}% of predictions.")