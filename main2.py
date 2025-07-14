import math
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt

path = "C:/Users/nsris/Downloads/gdp-vs-happiness/gdp-vs-happiness.csv"
df = pd.read_csv(path)

df_cleaned = df.dropna()
df_uncleaned = df[df.isna().any(axis=1)]
# print(df.columns)
# print(df_uncleaned.head())

X = df_cleaned[['GDP per capita, PPP (constant 2021 international $)']].values
Y = df_cleaned[['Cantril ladder score']].values

# plotting of data
plt.scatter(X, Y)
# plt.figure(figsize=(10,10))
# plt.xticks(np.arange(20000,100000, 5000))
plt.show()

model2 = KNeighborsRegressor(n_neighbors=3)
model2.fit(X, Y)
model = LinearRegression()
model.fit(X, Y)

pred1 = model.predict([[3046.5789]])
print("prediction from linear regression: ",str(pred1[0][0]))

pred2 = model2.predict([[3046.5789]])
print("prediction from K-nearest neighboours: ",str(pred2[0][0]))

slope = model.coef_[0][0]
intercept = model.intercept_[0]

print(slope, intercept)

# for all countries whose happiness index is missing
for ele in df_uncleaned[['GDP per capita, PPP (constant 2021 international $)']].values:
    if not math.isnan(ele[0]):
        print(str(ele) + " "+ str(model.predict([ele])))
