import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from util import im2col

x = np.random.rand(10, 1, 28, 28)
print(x.shape)
# print(x)
# %%

# %%
x1 = np.random.rand(1, 3, 7, 7)
col1 = im2col(x1, 5, 5, stride=1, pad=0)
print(col1.shape)  # (9, 75)
# %%
x2 = np.random.rand(10, 3, 7, 7)  # 10個數據
col2 = im2col(x2, 5, 5, stride=1, pad=0)
print(col2.shape)  # (90, 75)

array = np.array([[1, 2, 3],
                  [4, 5, 6]])
# %%
print("shape", array.shape)
print(array)

left = pd.DataFrame({'A': ['A0', 'A1', 'A2'],
                     'B': ['B0', 'B1', 'B2']}, index=['K0', 'K1', 'K2'])
right = pd.DataFrame({'C': ['C0', 'C2', 'C3'],
                      'D': ['D0', 'D2', 'D3']}, index=['K0', 'K2', 'K3'])

result = pd.merge(left, right, on=['key1', 'key2'], how='right')
print(result)

df1 = pd.DataFrame({'col1': [0, 1], 'col2': [4, 7]})
df2 = pd.DataFrame({'col1': [1, 2, 2], 'col2': [2, 2, 2]})
print(df1)
print(df2)
# %%
df3 = pd.concat([df1, df2], ignore_index=True)
print(df3)
# %%
result = pd.merge(df1, df2, on='col1', how='outer', indicator="indicator_column")
print(result)
print(left)
print(right)
# %%
result = pd.merge(left, right, left_index=True, right_index=True, how='outer')
print(result)
# %%
boys = pd.DataFrame({'k': ['K0', 'K1', 'K2'], 'age': [1, 2, 3]})
girls = pd.DataFrame({'k': ['K0', 'K0', 'K3'], 'age': [4, 5, 6]})
print(boys)
print(girls)
# %%
result = pd.merge(boys, girls, on='k', suffixes=['_boy', '_girl'], how='inner')
print(result)
# %%

# %%
data = pd.Series(np.random.randn(1000), index=np.arange(1000))
data = data.cumsum()
# %%
data = pd.DataFrame(np.random.randn(1000, 4), index=np.arange(1000), columns=list("ABCD"))
data = data.cumsum()
print(data.head())
# %%
data.plot()
plt.show()
