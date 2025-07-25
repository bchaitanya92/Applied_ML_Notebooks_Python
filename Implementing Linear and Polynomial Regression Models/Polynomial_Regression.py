import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')
sns.get_dataset_names()
data = sns.load_dataset('mpg')
data.head()
data.shape
data.info()
data.unique()
data['horsepower'].unique()
data.isnull().sum()
data.duplicated().sum()
df = data.copy()
df['horsepower'] = df['horsepower'].fillna(df['horsepower'].median(), inplace=
True)

df.describe()

numerical = df.select_dtypes(include=['int', 'float']).columns
categorical = df.select_dtypes(include=['object']).columns

print(numerical)
print(categorical)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')
sns.get_dataset_names()
data = sns.load_dataset('mpg')
data.head()
data.shape
data.info()
data.unique()
data['horsepower'].unique()
data.isnull().sum()
data.duplicated().sum()
df = data.copy()
df['horsepower'] = df['horsepower'].fillna(df['horsepower'].median(), inplace=
True)

df.describe().T

numerical = df.select_dtypes(include=['int', 'float']).columns
categorical = df.select_dtypes(include=['object']).columns

for col in numerical:
    plt.figure(figsize=(20, 4))
    plt.subplot(1, 2, 1)
    plt.hist(df[col], bins=20, alpha=0.5, color='g', edgecolor='black')
    plt.title(f'Histogram of {col}')
    plt.xlabel('')
    plt.ylabel('Frequency')
    plt.subplot(1, 2, 2)
    plt.boxplot(df[col], vert=False)
    plt.title(f'Boxplot of {col}')
    plt.tight_layout()
    plt.show()