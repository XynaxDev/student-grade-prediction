import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

df = pd.read_csv('data/student_data_cleaned.csv')

X = df[['Study Hours','Attendance (%)']]
y = df['Grades']


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 11)
grade_prediction_model = LinearRegression()
grade_prediction_model.fit(X_train,y_train)

with open('models/trained_model.pkl', 'wb') as f:
    pickle.dump(grade_prediction_model, f)

X_test.to_csv('data/X_test.csv', index=False)
y_test.to_csv('data/y_test.csv', index=False)

print("Model trained and saved to models/trained_model.pkl")
print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")