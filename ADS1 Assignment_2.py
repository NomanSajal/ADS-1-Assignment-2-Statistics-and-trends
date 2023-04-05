# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 20:09:09 2023

@author: Noman
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def read_data(filename, countries, indicator):
    """
    Reads a CSV file with World Bank data and extracts the values for the specified countries and indicator.

    Args:
        filename (str): the path to the CSV file.
        countries (List[str]): a list of country names to extract data for.
        indicator (str): the name of the indicator to extract data for.

    Returns:
        A tuple of two pandas DataFrames:
        - The first DataFrame contains the data for the specified countries and indicator, with years as rows and countries as columns.
        - The second DataFrame contains the same data, but with countries as rows and years as columns.
    """

    # Read in the data, excluding the "Unnamed: 66" column
    df = pd.read_csv(filename, skiprows=3, usecols=lambda col: col != 'Unnamed: 66')

    # Clean up column names
    df.columns = [str(col).strip().replace('\n', '') for col in df.columns]

    # Extract the desired countries and indicator
    df = df[df["Country Name"].isin(countries)]
    df = df[df["Indicator Name"] == indicator]

    # Separate the data into years and countries dataframes
    df_years = df.drop(["Country Name", "Country Code", "Indicator Name", "Indicator Code"], axis=1)
    df_years = df_years.transpose()
    df_years.columns = df_years.iloc[0]
    df_years = df_years[1:]
    df_years.index = pd.to_datetime(df_years.index)
    df_years.index.name = 'Year'
    df_years.columns.name = 'Country'

    #Transpose
    df_countries = df_years.transpose()
    df_countries.index = countries

    # Clean up the data types
    df_years = df_years.astype(float)
    df_countries = df_countries.astype(float)

    return df_years, df_countries


def plot_data(df: pd.DataFrame, years: list, y_label: str, title: str) -> None:
    """
    Plots a bar chart of the specified data for the specified years.

    Args:
        df (pd.DataFrame): the data to plot, with countries as rows and years as columns.
        years (list): a list of years to plot.
        y_label (str): the label for the y-axis.
        title (str): the title of the plot.

    Returns:
        None
    """

    # Subset the data to the specified years
    df_subset = df[years]

    # Set the index to the country names
    df_subset.index.name = "Country"
    df_subset.reset_index(inplace=True)
    df_subset.set_index("Country", inplace=True)

    # Plot the data
    ax = df_subset.plot(kind="bar", width=0.8)

    # Set the x-axis tick positions and labels
    ax.set_xticks(range(len(df_subset)))
    ax.set_xticklabels(df_subset.index)

    # Rotate the x-axis tick labels
    for tick in ax.get_xticklabels():
        tick.set_rotation(0)

    # Set the axis labels and legend
    ax.set_xlabel("Countries")
    ax.set_ylabel(y_label)
    ax.legend(title="Year")

    # Modify the legend labels and size
    handles, labels = ax.get_legend_handles_labels()
    labels = [label[:4] for label in labels]
    legend = ax.legend(handles, labels, title="Year", prop={'size': 8}, bbox_to_anchor=(1.02, 1))
    legend.get_frame().set_edgecolor('white')

    # Set the plot title  display and save the plot
    plt.title(title, fontsize=12)
    plt.savefig("bar_chart.png")

    plt.show()


filename = "worldbankdata.csv"
years = ["2000", "2005", "2010", "2015", "2020"]
countries = ["Bangladesh", "India", "Sri Lanka", "Nepal", "Pakistan"]

# Arable land (% of land area)
indicator = "Arable land (% of land area)"
df_years, df_countries = read_data(filename, countries, indicator)
plot_data(df_countries, years, "Arable land (% of land area)","Arable land (% of land area) by country and year")
plt.savefig("arable_land_bar_chart.png")
plt.show()
print(df_countries)


# Forest area (% of land area)
indicator = "Forest area (% of land area)"
df_years, df_countries = read_data(filename, countries, indicator)
plot_data(df_countries, years, "Forest area (% of land area)","Forest area (% of land area) by country and year")
plt.savefig("forest_area_bar_chart.png")
plt.show()

# Subset the data for the "Arable land (% of land area)" indicator and the specified years
df_arable_land = df_countries.loc[:, years]

# Use describe() method to get summary statistics
summary_arable_land = df_arable_land.describe()
print("Summary statistics for Arable land (% of land area):")
print(summary_arable_land)

# Calculate the coefficient of variation (CV) for each country
cv_arable_land = df_arable_land.std() / df_arable_land.mean()
print("Coefficient of variation for Arable land (% of land area):")
cv_arable_land.index = cv_arable_land.index.year
print(cv_arable_land)

# Read in the data for both indicators
indicator1 = "Arable land (% of land area)"
df_years1, df_countries1 = read_data(filename, countries, indicator1)


indicator2 = "Forest area (% of land area)"
df_years2, df_countries2 = read_data(filename, countries, indicator2)

# Subset the data to the specified years
years = ["2000", "2005", "2010", "2015", "2020"]
df1 = df_countries1.loc[:, years]
df2 = df_countries2.loc[:, years]

# Calculate the correlation coefficient
correlation_coefficient = df1.corrwith(df2).values[0]

print(f"Correlation coefficient between {indicator1} and {indicator2}: {correlation_coefficient:.2f}")

# Subset the data for the "Forest area (% of land area)" indicator and the specified years
df_forest_area = df_countries.loc[:, years]

# Use describe() method to get summary statistics
summary_forest_area = df_forest_area.describe()
print("Summary statistics for Forest area (% of land area):")
print(summary_forest_area)

# Calculate the coefficient of variation (CV) for each country
cv_forest_area = df_forest_area.std() / df_forest_area.mean()
print("Coefficient of variation for Forest area (% of land area):")
cv_forest_area.index = cv_forest_area.index.year
print(cv_forest_area)

# Urban population growth (annual %)"
indicator = "Urban population growth (annual %)"
df_years, df_countries = read_data(filename, countries, indicator)

# Subset the data for the specified years
population_growth = df_countries.loc[:, years]

# Create a line plot
plt.figure(figsize=(10, 6))
sns.set_style("darkgrid")
sns.lineplot(data=population_growth.T, palette="tab10", linewidth=3.5)
plt.title('Urban population growth (annual %) by country and year ',size=20)
plt.xlabel("Year")
plt.ylabel("(annual%) growth", size=20)
plt.legend(title="Country")
plt.savefig("line_plot.png")
plt.show()

# Scatter plot
fig, ax = plt.subplots()
ax.scatter(df1, df2)
ax.set_xlabel(indicator1)
ax.set_ylabel(indicator2)
ax.set_title(f"Correlation coefficient = {correlation_coefficient :.2f}")
plt.savefig("scatter_plot.png")
plt.show()