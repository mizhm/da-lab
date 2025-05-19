import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error


def read_file(path):
    try:
        df = pd.read_csv(path)
        if not df.empty:
            print(df.head(5))
            return df
        print('Dataframe is empty')
        return None
    except FileNotFoundError:
        print('File not found')
        return None


df = read_file('banking.txt')
print(df.info())

# Preprocessing
df_num = df.select_dtypes(include=[np.number])
df_num = df_num.fillna(df_num.mean())
print(df_num.info())


#Cau 1: Ve bieu do scatter
# df.plot.scatter(x='age', y='duration', marker='o', color='red')
# plt.title('Bieu do mo ta su phu thuoc cua duration vao age')
# plt.xlabel('Age')
# plt.ylabel('Duration')
# plt.show()


#Cau2: single feature linear regression model
linear_model = LinearRegression()
x = np.array(df_num['age']).reshape(-1, 1)
x = StandardScaler().fit_transform(x)
y = df_num['duration']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

linear_model.fit(X_train, y_train)
print('Phuong trinh hoi quy: ')
print('y = ', end='')
for i, item in enumerate(linear_model.coef_):
    print(f'{item:.2f}x{i + 1} + ', end='')
print(f'{linear_model.intercept_:.2f}')
print('R2 score: ', linear_model.score(X_test, y_test))
print('RMSE score: ', root_mean_squared_error(y_test, linear_model.predict(X_test)))


#Cau 3: multi features linear regression model
multi_model = LinearRegression()
X = np.array(df_num[['age', 'campaign']])
X = StandardScaler().fit_transform(X)
y = df_num['duration']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

multi_model.fit(X_train, y_train)
print('Phuong trinh hoi quy: ')
print('y = ', end='')
for i, item in enumerate(multi_model.coef_):
    print(f'{item:.2f}x{i + 1} + ', end='')
print(f'{multi_model.intercept_:.2f}')
print('R2 score của mô hình tren tap test: ', multi_model.score(X_test, y_test))


#Ve bieu do nhiet
matt_corr = df_num.corr()
matt_corr.dropna(inplace=True)
matt_corr.dropna(axis=1, inplace=True)
plt.figure(figsize=[12, 8])
sns.heatmap(matt_corr, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Ma tran tuong quan')
plt.show()