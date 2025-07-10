# linear regression for gdp per capita and happiness index
import numpy as np
import pandas as pd
from pandas.core.interchange.dataframe_protocol import DataFrame
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

path = "C:/Users/nsris/Downloads/gdp-vs-happiness(in).csv"
df = pd.read_csv(path)
df_cleaned = df.dropna()
df_uncleaned = df[df.isna().any(axis=1)]

# Afghanistan	2013		3046.5798
# print(df_uncleaned)
# print("------------------------------------------------------------------------------")
# print(df_cleaned)

# use cleaned data to model regression
years = df_uncleaned['Year'].values
df_uncleaned.dropna()
# print(years)
x = df_cleaned[['GDP per capita']].values
y = df_cleaned[['Cantril ladder score']].values

print("model trained assuming a linear relation b/w happiness index and GDP per capita")
plt.scatter(x, y)
plt.show()
model = LinearRegression()
model.fit(x, y)

num = 0
for ele in df_uncleaned['GDP per capita'].values:
 pred = model.predict([[ele]])
 print("for year "+str(years[num])+"predicted happiness is "+str(pred))
 num += 1


