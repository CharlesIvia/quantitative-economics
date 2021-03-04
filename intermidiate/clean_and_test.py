from math import log
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
from statsmodels.stats.diagnostic import normal_ad
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson

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

# Read data and clean it
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

# Deal with missing value in log_real interest rate
log_df = log_df.fillna(method="bfill")
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

# Kurtosis in the log df
log_df_kurtosis = log_df.kurtosis()
print(f"Log_df kurtosis (excess kurtosis): \n {log_df_kurtosis}")

print("=================================================")

# Jack-Bera test for the original df

print("Jack-Bera test for the original df")

j_b_ouput = ["j_b_statistic", "pvalue", "skewness", "kurtosis"]
j_b_real_gdp = jarque_bera(df["real_gdp"])
j_b_real_pop_growth = jarque_bera(df["pop_growth"])
j_b_real_exc_rate = jarque_bera(df["exc_rate"])
j_b_inflation = jarque_bera(df["inflation"])
j_b_real_real_interest = jarque_bera(df["real_interest"])
j_b_real_volatility = jarque_bera(df["volatility"])
j_b_Md = jarque_bera(df["Md"])

print(f"J-B for Md: {dict(zip(j_b_ouput, j_b_Md))}")
print(f"J-B for real gdp: {dict(zip(j_b_ouput, j_b_real_gdp))}")
print(f"J-B for pop growth: {dict(zip(j_b_ouput, j_b_real_pop_growth))}")
print(f"J-B for exc rate: {dict(zip(j_b_ouput, j_b_real_exc_rate))}")
print(f"J-B for inflation: {dict(zip(j_b_ouput, j_b_inflation))}")
print(f"J-B for real interest: {dict(zip(j_b_ouput, j_b_real_real_interest))}")
print(f"J-B for volatility: {dict(zip(j_b_ouput, j_b_real_volatility))}")

print("=================================================")


# Jack-Bera test for the log df

print("Jack-Bera test for the log df")

j_b_ouput = ["j_b_statistic", "pvalue", "skewness", "kurtosis"]
j_b_log_Md = jarque_bera(log_df["log_md"])
j_b_log_real_gdp = jarque_bera(log_df["log_real_gdp"])
j_b_log_real_pop_growth = jarque_bera(log_df["log_pop_growth"])
j_b_log_real_exc_rate = jarque_bera(log_df["log_exc_rate"])
j_b_log_inflation = jarque_bera(log_df["log_inflation"])
j_b_log_real_real_interest = jarque_bera(log_df["log_real_interest"])
j_b_log_real_volatility = jarque_bera(log_df["log_volatility"])


print(f"J-B for log Md: {dict(zip(j_b_ouput, j_b_log_Md))}")
print(f"J-B for log real gdp: {dict(zip(j_b_ouput, j_b_log_real_gdp))}")
print(f"J-B for log pop growth: {dict(zip(j_b_ouput, j_b_log_real_pop_growth))}")
print(f"J-B for log exc rate: {dict(zip(j_b_ouput, j_b_log_real_exc_rate))}")
print(f"J-B for log inflation: {dict(zip(j_b_ouput, j_b_log_inflation))}")
print(f"J-B for log real interest: {dict(zip(j_b_ouput, j_b_log_real_real_interest))}")
print(f"J-B for log volatility: {dict(zip(j_b_ouput, j_b_log_real_volatility))}")

print("=================================================")


# Distribution of variables


def show_distribution(dataframe, column, color):
    fig, axes = plt.subplots()
    sns.distplot(dataframe[column], color=color)
    plt.title(f"Normal distribution of {column}")
    plt.show()


# show_distribution(log_df, "log_md", "red")
# show_distribution(log_df, "log_real_gdp", "green")
# show_distribution(log_df, "log_pop_growth", "gold")
# show_distribution(log_df, "log_exc_rate", "purple")
# show_distribution(log_df, "log_inflation", "dodgerblue")
# show_distribution(log_df, "log_real_interest", "deeppink")
# show_distribution(log_df, "log_volatility", "yellow")


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

# Log df correlation

log_df_corr = log_df.corr()
print(f"Log df correlation matrix: \n {log_df_corr}")

print("=================================================")

# Linear regression

# Linear regression for original df

X = df.drop(["Md"], axis=1).copy()
X_log = log_df.drop(["log_md"], axis=1).copy()
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

df_lr_prediction = beta_0 + beta_1 * 2.252822
print(f"Prediction: \n {df_lr_prediction}")

print("=================================================")

# log_md  model (original df) using only volatility as a predictor

log_model = linear_model.LinearRegression()
log_model.fit(X_log[["log_volatility"]], y_cont)

beta_0 = log_model.intercept_
beta_1 = log_model.coef_[0]

print("The log model using only volatility as predictor variable: ")
print(f"Fit model: log (md) = {beta_0:.4f} + {beta_1:.4f} volatility")

log_model_prediction = beta_0 + beta_1 * 2.252822
print(f"Prediction: \n {log_model_prediction}")

print("=================================================")

# Full linear model using all the features- This model uses original df

print("Full linear model using all the features-uses original df:")

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
df_r_squared = df_full_lr_model.score(X, y)
print(f"R-squared for df model is: {df_r_squared}")

print("=================================================")

# Full linear model using all features (log md is used)

print("Full linear model using all features (log md is used):")

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

control_r_squared = control_df_full_lr_model.score(X, y_cont)
print(f"R-squared for control model is: {control_r_squared}")

print("=================================================")

# Full linear model using the log df

print("Full linear model using the log df:")

log_df_full_lr_model = linear_model.LinearRegression()
log_df_full_lr_model.fit(X_log, y_cont)

beta_0 = log_df_full_lr_model.intercept_
beta_1 = log_df_full_lr_model.coef_[0]
beta_2 = log_df_full_lr_model.coef_[1]
beta_3 = log_df_full_lr_model.coef_[2]
beta_4 = log_df_full_lr_model.coef_[3]
beta_5 = log_df_full_lr_model.coef_[4]
beta_6 = log_df_full_lr_model.coef_[5]

features = X_log.columns
coefs = list([beta_1, beta_2, beta_3, beta_4, beta_5, beta_6])
print(f"Model features: \n {features}")

coefs_dict = dict(zip(features, coefs))
print(f"Coefficeints dictionary: \n {coefs_dict}")

print(
    f"Fit model: log (md) = {beta_0:.4f} {beta_1:.15f} real_gdp {beta_2:.4f} pop_growth + {beta_3:.4f} exc_rate + {beta_4:.4f} inflation + {beta_5:.4f} real_interest {beta_6:.4f} volatility"
)

log_df["predicted_md"] = log_df_full_lr_model.predict(X_log)
log_df["residuals"] = abs(log_df["log_md"]) - abs(log_df["predicted_md"])

print(log_df)

log_r_squared = np.exp(control_df_full_lr_model.score(X_log, y_cont))
print(f"R-squared for full log model is: {log_r_squared}")

# Testing Linear regression assumptions

##Linearlity assumption


def linear_assumption(dataframe, actual):
    # Plotting the actual vs predicted values

    sns.lmplot(
        x=actual, y="predicted_md", data=dataframe, fit_reg=False, height=6, aspect=2
    )
    # Plotting the diagonal line

    line_df = pd.DataFrame()
    line_df["actual"] = dataframe["Md"]
    line_df["predicted"] = dataframe["predicted_md"]

    line_coords = np.arange(line_df.min().min(), line_df.max().max())
    plt.plot(
        line_coords, line_coords, color="darkorange", linestyle="--"  # X and y points
    )
    plt.title("Actual_Md vs. Predicted_Md")
    plt.tight_layout()
    plt.show()


linear_assumption(control_df, "Md")

##Normality of error terms


def normal_errors_assumption(dataframe, color, p_value_thresh=0.05):
    residuals = dataframe["residuals"]

    print("Using the Anderson-Darling test for normal distribution")

    p_value = normal_ad(residuals)[1]
    print("p-value from the test - below 0.05 generally means non-normal:", p_value)

    if p_value < p_value_thresh:
        print("Residuals are not normally distributed")
    else:
        print("Residuals are normally distributed")

    # Plotting the residuals distribution
    sns.distplot(dataframe["residuals"], color=color)
    plt.title(f"Normal distribution of residuals")
    plt.show()

    print()
    if p_value > p_value_thresh:
        print("Assumption satisfied")
    else:
        print("Assumption not satisfied")
        print()
        print("Confidence intervals will likely be affected")
        print("Try performing nonlinear transformations on variables")


normal_errors_assumption(control_df, "blue")
normal_errors_assumption(log_df, "blue")

## Little to no multicollinearlity


def multicollinearlity_assumption(features, feature_names=None):
    print("Assumption 3: Little to no multicollinearity among predictors")

    # Plotting the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(pd.DataFrame(features, columns=feature_names).corr(), annot=True)
    plt.title("Correlation of Variables")
    plt.show()


multicollinearlity_assumption(X)

## No Autocorrelation of the error terms


def autocorrelation_assumption(dataframe):
    print("Assumption 4: No Autocorrelation", "\n")

    residuals = dataframe["residuals"]
    print("\nPerforming Durbin-Watson Test")
    print(
        "Values of 1.5 < d < 2.5 generally show that there is no autocorrelation in the data"
    )
    print("0 to 2< is positive autocorrelation")
    print(">2 to 4 is negative autocorrelation")
    print("-------------------------------------")

    durbinWatson = durbin_watson(residuals)
    print("Durbin-Watson:", durbinWatson)
    if durbinWatson < 1.5:
        print("Signs of positive autocorrelation", "\n")
        print("Assumption not satisfied")
    elif durbinWatson > 2.5:
        print("Signs of negative autocorrelation", "\n")
        print("Assumption not satisfied")
    else:
        print("Little to no autocorrelation", "\n")
        print("Assumption satisfied")


autocorrelation_assumption(control_df)
autocorrelation_assumption(log_df)

# Homoscedasticity


def homoscedasticity_assumption(dataframe):
    # Calculating residuals for the plot
    residuals = dataframe["residuals"]
    dataframe["ind"] = range(0, 40, 1)

    # Plotting the residuals
    plt.subplots(figsize=(12, 6))
    ax = plt.subplot(111)  # To remove spines
    plt.scatter(x=dataframe.ind, y=residuals, alpha=0.5)
    plt.plot(np.repeat(0, dataframe.ind.max()), color="darkorange", linestyle="--")
    ax.spines["right"].set_visible(False)  # Removing the right spine
    ax.spines["top"].set_visible(False)  # Removing the top spine
    plt.title("Residuals")
    plt.show()


homoscedasticity_assumption(control_df)
homoscedasticity_assumption(log_df)
