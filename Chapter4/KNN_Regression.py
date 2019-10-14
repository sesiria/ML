import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# read data
df = pd.read_csv('data.csv')
df # data frame

# feature processing
# one hot endocding for 'Color
df_colors = df['Color'].str.get_dummies().add_prefix('Color: ')
# one hot encoding for 'Type'
df_type = df['Type'].apply(str).str.get_dummies().add_prefix('Type: ')
# add on hot encoding column
df = pd.concat([df, df_colors, df_type], axis = 1)
# remove the original column before the one hot encoding.
df = df.drop(['Brand', 'Type', 'Color'], axis = 1)

df

# data convert
matrix = df.corr()
f, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(matrix, square=True)
plt.title('Car Price Variables')


from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import numpy as np

# get input data X and lablels Y
X = df[['Construction Year', 'Days Until MOT', 'Odometer']]
y = df['Ask Price'].values.reshape(-1, 1)
# split train, test data set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 41)

# data normalization
X_normalizer = StandardScaler() # N(0, 1)
X_train = X_normalizer.fit_transform(X_train)
X_test = X_normalizer.transform(X_test)

y_normalizer = StandardScaler()
y_train = y_normalizer.fit_transform(y_train)
y_test = y_normalizer.transform(y_test)

knn = KNeighborsRegressor(n_heighbors = 2)
knn.fit(X_train, y_train.ravel())

# Now we can predict prices:
y_pred = knn.predict(X_test)
y_pred_inv = y_normalizer.inverse_transform(y_pred)
y_test_inv = y_normalizer.inverse_transform(y_test)

# Build a plot
plt.scatter(y_pred_inv, y_test_inv)
plt.xlabel('Prediction')
plt.ylabel('Real value')

# Now add the perfect prediction line
diagonal = np.linspace(500, 1500, 100)
plt.plot(diagonal, diagonal, '-r')
plt.xlabel('Predicted ask price')
plt.ylabel('Ask price')
plt.show()