import pandas as pd
import matplotlib.pyplot as plt

# load the dataset into a pandas DataFrame
df = pd.read_csv('dataset.csv')

# print the number of rows and columns in the dataset
print('Number of rows:', df.shape[0])
print('Number of columns:', df.shape[1])

# print the first five rows of the dataset
print(df.head())

# print the summary statistics of the dataset
print(df.describe())

# calculate the skewness and kurtosis of each column
skewness = df.skew()
kurtosis = df.kurt()

# print the skewness and kurtosis of each column
print('Skewness:')
print(skewness)
print('Kurtosis:')
print(kurtosis)

# plot a histogram of each column
for column in df.columns:
    plt.hist(df[column])
    plt.title(column)
    plt.show()

# plot a scatter plot of each pair of columns
for i in range(df.shape[1]-1):
    for j in range(i+1, df.shape[1]):
        plt.scatter(df.iloc[:,i], df.iloc[:,j])
        plt.xlabel(df.columns[i])
        plt.ylabel(df.columns[j])
        plt.show()
