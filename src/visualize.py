import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

df = pd.read_csv('data/student_data_cleaned.csv')
with open('models/trained_model.pkl', 'rb') as file:
    grade_prediction_model = pickle.load(file)

# study Hours after removing outliers
sns.boxplot(data=df, x='Study Hours')
plt.title('Study Hours without outliers')

# Plotting bar graph to visualize the predictions of grades
X_test = df[['Study Hours','Attendance (%)']]
y_test = df['Grades']

y_pred = grade_prediction_model.predict(X_test)
n_samples = 10
y_test_sample = y_test[:n_samples]
y_pred_sample = y_pred[:n_samples]

plt.figure(figsize=(13, 6))
x = np.arange(n_samples)
plt.bar(x - 0.2, y_test_sample, 0.4, label='Actual Grades', color='green')
plt.bar(x + 0.2, y_pred_sample, 0.4, label='Predicted Grades', color='orange')
plt.xlabel('Student Number')
plt.ylabel('Grades')
plt.title('Actual vs Predicted Grades (First 10 Students)')
plt.xticks(x, [f'Student {i+1}' for i in x])
plt.legend()
plt.show()