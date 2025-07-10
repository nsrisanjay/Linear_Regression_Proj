import math
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = "C:/Users/nsris/Downloads/gdp-vs-happiness/gdp-vs-happiness.csv"
df = pd.read_csv(path)

df_cleaned = df.dropna()
df_uncleaned = df[df.isna().any(axis=1)]
# print(df.columns)
# print(df_uncleaned.head())

X = df_cleaned[['GDP per capita, PPP (constant 2021 international $)']].values
Y = df_cleaned[['Cantril ladder score']].values
plt.scatter(X, Y)
plt.show()

model = LinearRegression()
model.fit(X, Y)

slope = model.coef_[0][0]
intercept = model.intercept_[0]

print(slope, intercept)

for ele in df_uncleaned[['GDP per capita, PPP (constant 2021 international $)']].values:
    if not math.isnan(ele[0]):
        print(str(ele) + " "+ str(model.predict([ele])))
