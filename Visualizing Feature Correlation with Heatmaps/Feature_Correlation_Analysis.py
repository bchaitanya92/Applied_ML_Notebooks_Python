import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['Target'] = data.target

variable_meaning = {
    "MedInc": "Median income in block group",
    "HouseAge": "Median house age in block group",
    "AveRooms": "Average number of rooms per household",
    "AveBedrms": "Average number of bedrooms per household",
    "Population": "Population of block group",
    "AveOccup": "Average number of household members",
    "Latitude": "Latitude of block group",
    "Longitude": "Longitude of block group",
    "Target": "Median house value ($100,000s)"
}
variable_df = pd.DataFrame(list(variable_meaning.items()), columns=["Feature", "Description"])

print("Variable Meaning Table:")
print(variable_df)
print("Basic Information about Dataset:")
print(df.info())
print("First 5 Rows of Dataset:")
print(df.head())
print("Summary Statistics:")
print(df.describe())
print("Missing values in each column:")
print(df.isnull().sum())

plt.figure(figsize=(12, 8))
df.hist(figsize=(12, 8), bins=30, edgecolor="black")
plt.suptitle("Feature Distributions", fontsize=16)
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(data=df)
plt.xticks(rotation=45)
plt.title("Box Plot of Features to Identify Outliers")
plt.show()

plt.figure(figsize=(10, 6))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Feature Correlation Heatmap")
plt.show()

sns.pairplot(df[['MedInc', 'HouseAge', 'AveRooms', 'Target']], diag_kind="kde")
plt.show()

print("Key Insights:")
print("1. The dataset has a shape of", df.shape[0], "rows &", df.shape[1], "columns.")
print("2. No missing values were found in the dataset.")
print("3. Histograms show skewed distributions in some features like 'MedInc'.")
print("4. Box plots indicate potential outliers in 'AveRooms' & 'AveOccup'.")
print("5. Correlation heatmap shows 'MedInc' has the highest correlation with house prices.")