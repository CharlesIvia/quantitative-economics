import wbgapi as wb
import xlsxwriter
import pandas as pd

# Function to mine data from the API


def get_values_1970_2019(indicator, country="KEN"):
    data = wb.data.DataFrame(indicator, country)
    data_T = data.T
    clean_data = data_T.dropna()
    data_1980_2019 = clean_data.loc["YR1970":"YR2019"]
    return data_1980_2019


# Indicator variables
inflation_rate_indicator = ["FP.CPI.TOTL.ZG"]
real_interest_indicator = ["FR.INR.RINR"]
official_exchange_rate_indicator = ["PA.NUS.FCRF"]
pop_growth_rate_indicator = ["SP.POP.GROW"]
real_gdp_indicator = ["NY.GDP.MKTP.CD"]
broad_money_pc_gdp_indicator = ["FM.LBL.BMNY.GD.ZS"]
population_indicator = ["SP.POP.TOTL"]
per_capita_USD_indicator = ["NY.GDP.PCAP.CD"]
gdp_growth_indicator = ["NY.GDP.MKTP.KD.ZG"]
lending_interest_indicator = ["FR.INR.LEND"]
deposit_interest_rate_indicator = ["FR.INR.DPST"]

# Output from the api
real_interest = get_values_1970_2019(real_interest_indicator)
inflation = get_values_1970_2019(inflation_rate_indicator)
ex_rate = get_values_1970_2019(official_exchange_rate_indicator)
pop_growth_rate = get_values_1970_2019(pop_growth_rate_indicator)
real_gdp = get_values_1970_2019(real_gdp_indicator)
broad_money = get_values_1970_2019(broad_money_pc_gdp_indicator)
pop = get_values_1970_2019(population_indicator)
per_capita = get_values_1970_2019(per_capita_USD_indicator)
gdp_growth = get_values_1970_2019(gdp_growth_indicator)
lending_interest = get_values_1970_2019(lending_interest_indicator)
deposit_rate = get_values_1970_2019(deposit_interest_rate_indicator)

# Create a dataframe

df = pd.DataFrame(pop)
df = df.rename(columns={"KEN": "population"})
df["broad_money"] = broad_money
df["real_gdp"] = real_gdp
df["pop_growth"] = pop_growth_rate
df["exc_rate"] = ex_rate
df["inflation"] = inflation
df["real_interest"] = real_interest
df["per_capita_USD"] = per_capita
df["gdp_growth_rate"] = gdp_growth
df["lending_interest_rate"] = lending_interest
df["deposit_interest_rate"] = deposit_rate

print(df)

# Create a pandas excel writer

writer = pd.ExcelWriter("project_data.xlsx", engine="xlsxwriter")

# Convert df to xlsxwriter Excel object

df.to_excel(writer, sheet_name="first")

# Close Pandas Excel Write and output the Excel file

writer.save()
