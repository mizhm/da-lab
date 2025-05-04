import numpy as np
import pandas as pd

# Cau 1
def read(path):
    try:
        df = pd.read_csv(path)
        if not df.empty:
            print(df.head())
            return df
        print('dataframe is empty')
    except FileNotFoundError:
        print('path not found')

df = read('./winequality-red.csv')


#Cau 2
def summarize(df):
    df_num = df.select_dtypes(include=[np.number])
    sum = df_num.agg(func=['min', 'max', 'mean', 'median', 'var', 'std'])
    sum.loc['q1'] = df_num.quantile(0.25)
    sum.loc['q2'] = df_num.quantile(0.5)
    sum.loc['q3'] = df_num.quantile(0.75)
    sum.loc['iqr'] = df_num.quantile(0.75) - df.quantile(0.25)
    print(sum)

summarize(df)


#Cau 3
import matplotlib.pyplot as plt
df.hist()
plt.show()



#Cau 4
df.plot.box()
plt.show()


#Cau 5
df_num = df.select_dtypes(include=[np.number])
df_num_3_col = df_num.iloc[:, :3]
print(df_num_3_col.describe())


#Cau 6
# Tính ma trận tương quan
correlation_matrix = df.corr()

# Lấy hàng tương quan với cột 'quality'
quality_correlation = correlation_matrix['quality'].sort_values(ascending=False)

# In ra hệ số tương quan với cột 'quality'
print("Hệ số tương quan với cột 'quality':")
print(quality_correlation)

# Xác định yếu tố có tương quan dương lớn nhất (ảnh hưởng tích cực nhất)
positive_influence = quality_correlation[quality_correlation < 1].idxmax()
positive_correlation_value = quality_correlation[positive_influence]
print(f"\nYếu tố có ảnh hưởng tích cực nhất đến chất lượng là '{positive_influence}' với hệ số tương quan là {positive_correlation_value:.3f}")

# Xác định yếu tố có tương quan âm lớn nhất (ảnh hưởng tiêu cực nhất)
negative_influence = quality_correlation.idxmin()
negative_correlation_value = quality_correlation[negative_influence]
print(f"Yếu tố có ảnh hưởng tiêu cực nhất đến chất lượng là '{negative_influence}' với hệ số tương quan là {negative_correlation_value:.3f}")

