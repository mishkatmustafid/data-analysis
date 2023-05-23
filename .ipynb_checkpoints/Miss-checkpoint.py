import pandas as pd

# Load data from a CSV file
data = pd.read_csv('data.csv')

# Check for incorrect data types
data['age'] = pd.to_numeric(data['age'], errors='coerce')
data['is_active'] = data['is_active'].astype(bool)

# Check for irrelevant variables and drop them
data = data.drop(columns=['id'])

# Check for missing values and fill them with mean or median
mean_age = data['age'].mean()
data['age'].fillna(mean_age, inplace=True)

median_income = data['income'].median()
data['income'].fillna(median_income, inplace=True)

# Check for duplicates and drop them
data.drop_duplicates(inplace=True)

# Print the cleaned data
print(data.head())
