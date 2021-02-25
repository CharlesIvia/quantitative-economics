import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.utils import axis0_safe_slice
from sklearn import (
    linear_model,
    metrics,
    neural_network,
    pipeline,
    model_selection,
    tree,
    neural_network,
    preprocessing,
)

# Read data
df = pd.read_excel("project_data.xlsx", index_col="Unnamed: 0")
print(df.columns)
columns_to_drop = [
    "population",
    "per_capita_USD",
    "gdp_growth_rate",
    "lending_interest_rate",
    "deposit_interest_rate",
    "current_exports",
    "modeled_unemp",
    "imports_in_USD",
    "cpi",
]

df = df.drop(columns_to_drop, axis=1)

df.rename(columns={"Unnamed: 0": "Year", "broad_money": "Md"}, inplace=True)

df = df.iloc[10:]

# volatility
real_interest = df["real_interest"]
rolling_std = real_interest.rolling(min_periods=1, window=40).std()
df["int_roll"] = rolling_std

print(df)

# Extract columns
columns = df.columns
print(columns)

# if mean > median, data is skewed to the right

# Skewness - How much a distibution leans on the left or right (-ve skew and +skew)
df_skewness = skew(df)
print(dict(zip(columns, df_skewness)))

# Kurtosis - measure of the thickness of tails of distribution (proxy for probability of outliers)
# Normal  = 3
# Excess kurtosis > 0 suggests non normality
# -ve excess kurtosis - platykurtic (tail thinner than normal distribution)
# +ve  exess kurtosis- leptokurtic (distibution is heavier than no-dist)

df_kurtosis = kurtosis(df)
print(dict(zip(columns, df_kurtosis)))

# Correlation

df_corr = df.corr()
print(df_corr)
# Show distribution


def show_distribution(column, color):
    fig, axes = plt.subplots()
    return sns.histplot(df[column], color=color)


# show_distribution("real Int", "deeppink")
# show_distribution("inflation", "dodgerblue")
# show_distribution("Md", "red")
# show_distribution("real GDP", "green")
# show_distribution("exc rate", "purple")
# show_distribution("pop grate", "gold")
# show_distribution("int_roll", "yellow")


def std_data(s):
    mu = s.mean()
    su = s.std()

    std_series = []
    for index, value in s.items():
        std_series.append((value - mu) / su)

    return std_series


df["std_Md"] = std_data(df["Md"])
df["std_gdp"] = std_data(df["real_gdp"])
df["std_real_int"] = std_data(df["real_interest"])
df["std_inflation"] = std_data(df["inflation"])
df["std_exc_rate"] = std_data(df["exc_rate"])
df["std_pop"] = std_data(df["pop_growth"])
df["std_int_roll"] = std_data(df["int_roll"])

# df.drop(columns, axis=1).plot()

std_df = df.drop(columns, axis=1)
std_cols = std_df.columns
df = df.drop(std_cols, axis=1)
print(std_df)
std_df.plot()

normalized_df = (df - df.mean()) / df.std()

# normalized_df.plot()
print(normalized_df)


# 2 variable regression

df = df.iloc[1:]

X = df.drop(["Md"], axis=1).copy()

for col in list(X):
    X[col] = X[col].astype(float)

print(X.head())

y = np.log(df["Md"])
df["log_md"] = y
print(y.head())
print(df)


def var_scatter(df, ax=None, var="int_roll"):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    df.plot.scatter(x=var, y="log_md", s=1.5, ax=ax)
    return ax


var_scatter(df)

sns.lmplot(data=df, x="int_roll", y="log_md", height=6, scatter_kws=dict(s=1.5))

# construct the model instance
md_lr_model = linear_model.LinearRegression()

# fir the model
md_lr_model.fit(X[["int_roll"]], y)

# print coefficients
beta_0 = md_lr_model.intercept_
beta_1 = md_lr_model.coef_[0]

print(f"Fit model: log (md) = {beta_0:.4f} + {beta_1:.4f} int_roll")

ax = var_scatter(df)
x = np.array([0, df["int_roll"].max()])
ax.plot(x, beta_0 + beta_1 * x)


logi_7 = md_lr_model.predict([[7]])[0]
print(
    f"The model predicts an interest vol of 7 would lead to a money demand of {np.exp(logi_7):.2f}"
)

# Multivariate regression

plt.show()
