import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from scipy import stats
from statsmodels.stats.stattools import jarque_bera
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.utils import axis0_safe_slice
import qeds

colors = qeds.themes.COLOR_CYCLE
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
df = df.iloc[9:]

# volatility
real_interest = df["real_interest"]
rolling_std = real_interest.rolling(min_periods=1, window=40).std()
df["volatility"] = rolling_std
df = df.iloc[1:]
print(df)

# Extract columns
columns = df.columns

# Change type

for col in list(df):
    df[col] = df[col].astype(float)

# Create log transformed df

log_df = pd.DataFrame()

log_df["log_md"] = np.log(df["Md"])
log_df["log_real_gdp"] = np.log(df["real_gdp"])
log_df["log_pop_growth"] = np.log(df["pop_growth"])
log_df["log_exc_rate"] = np.log(df["exc_rate"])
log_df["log_inflation"] = np.log(df["inflation"])
log_df["log_real_interest"] = np.log(df["real_interest"])
log_df["log_volatility"] = np.log(df["volatility"])

print(log_df)

print("=================================================")

# Skewness in original df
df_skew = df.skew()
print(f"Df_Skew skewness: \n {df_skew}")

print("=================================================")

# Skewness in the log-transformed df
log_df_skew = log_df.skew()
print(f"Log_df skewness: \n {log_df_skew}")

print("=================================================")

# Creating a normal df
normal_df = pd.DataFrame()
normal_df["Md"] = df["Md"]
normal_df["real_gdp"] = log_df["log_real_gdp"]
normal_df["pop_growth"] = log_df["log_pop_growth"]
normal_df["exc_rate"] = df["exc_rate"]
normal_df["inflation"] = log_df["log_inflation"]
normal_df["real_interest"] = df["real_interest"]
normal_df["volatility"] = df["volatility"]

print(normal_df)

print("=================================================")

# Skewness in normal df
normal_df_skew = normal_df.skew()
print(f"Normal_df skewness: \n {normal_df_skew}")

print("=================================================")

# Kurtosis in the normal df
normal_df_kurtosis = normal_df.kurtosis()
print(f"Normal_df kurtosis: \n {normal_df_kurtosis}")

print("=================================================")

# Jack-Bera test

j_b_ouput = ["j_b_statistic", "pvalue", "skewness", "kurtosis"]
j_b_real_gdp = jarque_bera(df["real_gdp"])
j_b_real_pop_growth = jarque_bera(df["pop_growth"])
j_b_real_exc_rate = jarque_bera(df["exc_rate"])
j_b_inflation = jarque_bera(df["inflation"])
j_b_real_real_interest = jarque_bera(df["real_interest"])
j_b_real_volatility = jarque_bera(df["volatility"])
j_b_Md = jarque_bera(df["Md"])

# Jack-bera for normal_df

# j_b_ouput = ["j_b_statistic", "pvalue", "skewness", "kurtosis"]
# j_b_real_gdp = jarque_bera(normal_df["real_gdp"])
# j_b_real_pop_growth = jarque_bera(normal_df["pop_growth"])
# j_b_real_exc_rate = jarque_bera(normal_df["exc_rate"])
# j_b_inflation = jarque_bera(normal_df["inflation"])
# j_b_real_real_interest = jarque_bera(normal_df["real_interest"])
# j_b_real_volatility = jarque_bera(normal_df["volatility"])
# j_b_Md = jarque_bera(normal_df["Md"])

print(f"J-B for Md: {dict(zip(j_b_ouput, j_b_Md))}")
print(f"J-B for real gdp: {dict(zip(j_b_ouput, j_b_real_gdp))}")
print(f"J-B for pop growth: {dict(zip(j_b_ouput, j_b_real_pop_growth))}")
print(f"J-B for exc rate: {dict(zip(j_b_ouput, j_b_real_exc_rate))}")
print(f"J-B for inflation: {dict(zip(j_b_ouput, j_b_inflation))}")
print(f"J-B for real interest: {dict(zip(j_b_ouput, j_b_real_real_interest))}")
print(f"J-B for volatility: {dict(zip(j_b_ouput, j_b_real_volatility))}")

print("=================================================")

# Distribution of variables


def show_distribution(column, color):
    fig, axes = plt.subplots()
    sns.distplot(normal_df[column], color=color)
    plt.title(f"Normal distribution of {column}")
    plt.show()


# show_distribution("Md", "red")
# show_distribution("real_gdp", "green")
# show_distribution("pop_growth", "gold")
# show_distribution("exc_rate", "purple")
# show_distribution("inflation", "dodgerblue")
# show_distribution("real_interest", "deeppink")
# show_distribution("volatility", "yellow")

# Show trends in variables

# Creatting a standardized df


def std_data(s):
    mu = s.mean()
    su = s.std()

    std_series = []
    for index, value in s.items():
        std_series.append((value - mu) / su)

    return std_series


std_df = pd.DataFrame()

std_df["Md"] = std_data(df["Md"])
std_df["real_gdp"] = std_data(df["real_gdp"])
std_df["pop_growth"] = std_data(df["pop_growth"])
std_df["exc_rate"] = std_data(df["exc_rate"])
std_df["inflation"] = std_data(df["inflation"])
std_df["real_interest"] = std_data(df["real_interest"])
std_df["volatility"] = std_data(df["volatility"])

print(std_df)

print("=================================================")


def visualize_trends():
    std_df.plot(figsize=(12, 6))
    plt.title("Variable Trends")
    plt.show()


visualize_trends()

# Correlation among variables

# Original df correlation

df_corr = df.corr()
print(f"Original df correlation matrix: \n {df_corr}")

print("=================================================")

# Normal df correlation

normal_df_corr = normal_df.corr()
print(f"Normal df correlation matrix: \n {normal_df_corr}")

print("=================================================")

# Linear regression

# Linear regression for original df

X = df.drop(["Md"], axis=1).copy()
X_norm = normal_df.drop(["Md"], axis=1).copy()

print(X.head())

print("=================================================")

y = df["Md"]
df["y_md"] = y
print(df)

y_cont = np.log(df["Md"])
control_df = df.copy()

print("=================================================")


# Scatter plot for df
def var_scatter(df, ax=None, var="volatility"):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    df.plot.scatter(x=var, y="y_md", s=1.5, ax=ax)
    plt.title("Scatter plot")
    return ax


var_scatter(df)


def show_lmplot():
    sns.lmplot(data=df, x="volatility", y="y_md", height=6, scatter_kws=dict(s=1.5))
    plt.tight_layout()
    plt.title("Lmplot")
    plt.show()


show_lmplot()

# Linear model for for the orginal df using only volatility as predictor variable

df_lr_model = linear_model.LinearRegression()
df_lr_model.fit(X[["volatility"]], y)

beta_0 = df_lr_model.intercept_
beta_1 = df_lr_model.coef_[0]

print("The df_lr model using only volatility as predictor variable: ")
print(f"Fit model: log (md) = {beta_0:.4f} + {beta_1:.4f} volatility")

print("=================================================")

# log_md  model (original df)

log_model = linear_model.LinearRegression()
log_model.fit(X[["volatility"]], y_cont)

beta_0 = log_model.intercept_
beta_1 = log_model.coef_[0]

print("The log model using only volatility as predictor variable: ")
print(f"Fit model: log (md) = {beta_0:.4f} + {beta_1:.4f} volatility")

print("=================================================")

# Full linear model using all the features

df_full_lr_model = linear_model.LinearRegression()
df_full_lr_model.fit(X, y)


beta_0 = df_full_lr_model.intercept_
beta_1 = df_full_lr_model.coef_[0]
beta_2 = df_full_lr_model.coef_[1]
beta_3 = df_full_lr_model.coef_[2]
beta_4 = df_full_lr_model.coef_[3]
beta_5 = df_full_lr_model.coef_[4]
beta_6 = df_full_lr_model.coef_[5]

features = X.columns
coefs = list([beta_1, beta_2, beta_3, beta_4, beta_5, beta_6])
print(f"Model features: \n {features}")

coefs_dict = dict(zip(features, coefs))
print(f"Coefficeints dictionary: \n {coefs_dict}")

print(
    f"Fit model: log (md) = {beta_0:.4f} {beta_1:.15f} real_gdp {beta_2:.4f} pop_growth + {beta_3:.4f} exc_rate + {beta_4:.4f} inflation + {beta_5:.4f} real_interest {beta_6:.4f} volatility"
)

df["predicted_md"] = df_full_lr_model.predict(X)
df["residuals"] = abs(df["Md"]) - abs(df["predicted_md"])

print(df)


print("=================================================")

# Full linear log model using all features

control_df_full_lr_model = linear_model.LinearRegression()
control_df_full_lr_model.fit(X, y_cont)


beta_0 = control_df_full_lr_model.intercept_
beta_1 = control_df_full_lr_model.coef_[0]
beta_2 = control_df_full_lr_model.coef_[1]
beta_3 = control_df_full_lr_model.coef_[2]
beta_4 = control_df_full_lr_model.coef_[3]
beta_5 = control_df_full_lr_model.coef_[4]
beta_6 = control_df_full_lr_model.coef_[5]

features = X.columns
coefs = list([beta_1, beta_2, beta_3, beta_4, beta_5, beta_6])
print(f"Model features: \n {features}")

coefs_dict = dict(zip(features, coefs))
print(f"Coefficeints dictionary: \n {coefs_dict}")

print(
    f"Fit model: log (md) = {beta_0:.4f} {beta_1:.15f} real_gdp {beta_2:.4f} pop_growth + {beta_3:.4f} exc_rate + {beta_4:.4f} inflation + {beta_5:.4f} real_interest {beta_6:.4f} volatility"
)

control_df["predicted_md"] = np.exp(control_df_full_lr_model.predict(X))
control_df["residuals"] = abs(control_df["Md"]) - abs(control_df["predicted_md"])

print(control_df)


print("=================================================")

# Full linear model using normal df

normal_df_full_lr_model = linear_model.LinearRegression()
normal_df_full_lr_model.fit(X_norm, y)

beta_0 = normal_df_full_lr_model.intercept_
beta_1 = normal_df_full_lr_model.coef_[0]
beta_2 = normal_df_full_lr_model.coef_[1]
beta_3 = normal_df_full_lr_model.coef_[2]
beta_4 = normal_df_full_lr_model.coef_[3]
beta_5 = normal_df_full_lr_model.coef_[4]
beta_6 = normal_df_full_lr_model.coef_[5]

features = X_norm.columns
coefs = list([beta_1, beta_2, beta_3, beta_4, beta_5, beta_6])
print(f"Model features: \n {features}")

coefs_dict = dict(zip(features, coefs))
print(f"Coefficeints dictionary: \n {coefs_dict}")

print(
    f"Fit model: log (md) = {beta_0:.4f} {beta_1:.15f} real_gdp {beta_2:.4f} pop_growth + {beta_3:.4f} exc_rate + {beta_4:.4f} inflation + {beta_5:.4f} real_interest {beta_6:.4f} volatility"
)

normal_df["predicted_md"] = normal_df_full_lr_model.predict(X_norm)
normal_df["residuals"] = abs(normal_df["Md"]) - abs(normal_df["predicted_md"])

print(normal_df)
