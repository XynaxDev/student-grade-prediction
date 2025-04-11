import numpy as np
import pandas as pd

def whisker(col):
    q1, q3 = np.percentile(col, [25,75])
    iqr = q3 - q1
    lower_whisker = q1 - 1.5 * iqr
    upper_whisker = q3 + 1.5 * iqr
    return lower_whisker, upper_whisker


df = pd.read_csv('data/student_data.csv')

# Now capping study hours outliers using whisker method
lower_whisker, upper_whisker = whisker(df['Study Hours'])
df['Study Hours'] = np.where(df['Study Hours'] < lower_whisker, lower_whisker, df['Study Hours'])
df['Study Hours'] = np.where(df['Study Hours'] > upper_whisker, upper_whisker, df['Study Hours'])

# saving cleaned data
df.to_csv('data/student_data_cleaned.csv', index=False)
print(f"Cleaned data saved to data/student_data_cleaned.csv")
print(f"Lower whisker: {lower_whisker:.2f}, Upper whisker: {upper_whisker:.2f}")

# Rows affected
rows_count = (df['Study Hours'] == lower_whisker).sum() + (df['Study Hours'] == upper_whisker).sum()
percent_of_rows = rows_count / df.shape[0] * 100
print(f"About {percent_of_rows:.2f} % of rows were affected!")
