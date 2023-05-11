# -*- coding: utf-8 -*-
"""
Created on Wed May 10 05:17:24 2023

@author: Anika
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
def plot_clustered_data(df, countries, indicators, n_clusters):
    
    """
    Plots a scatter plot of the selected countries and indicators, clustered by KMeans into n_clusters.

    Parameters:
        df (pandas.DataFrame): the DataFrame containing the data
        countries (list): a list of the countries to include in the plot
        indicators (list): a list of the indicators to include in the plot
        n_clusters (int): the number of clusters to form

    Returns:
        None
    """
    
    
    # Extract the desired countries and indicators
    df = df[df["Country Name"].isin(countries)]
    df = df[df["Indicator Name"].isin(indicators)]

    # Filter for the years 1980 to 2020
    df = df[['Country Name', '1980', '1985','1990','1995', '2000', '2005', '2010', '2015', '2020']]

    # Set the index to the country names
    df = df.set_index('Country Name')

    # Convert the data to a numpy array
    data = df.to_numpy()

    # Create a KMeans clustering object with n_clusters clusters
    kmeans = KMeans(n_clusters=n_clusters)

    # Fit the KMeans object to the data
    kmeans.fit(data)

    # Get the cluster labels and centers
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
   

    # Plot the data and cluster centers
    colors = ['b', 'g', 'r', 'c', 'y'] # Set colors for each country
    fig, axs = plt.subplots(1, n_clusters, figsize=(20,5))

    for i, country in enumerate(countries):
        for ax in axs:
            ax.scatter(data[i, 0], data[i, 1], c=colors[i], label=country, s=80)

    for i in range(len(data)):
        if labels[i] == 0:
            axs[0].scatter(data[i, 0], data[i, 1], c=colors[labels[i]])
        elif labels[i] == 1:
            axs[1].scatter(data[i, 0], data[i, 1], c=colors[labels[i]])
        elif labels[i] == 2:
            axs[2].scatter(data[i, 0], data[i, 1], c=colors[labels[i]])
        elif labels[i] == 3:
            axs[3].scatter(data[i, 0], data[i, 1], c=colors[labels[i]])
        elif labels[i] == 4:
            axs[4].scatter(data[i, 0], data[i, 1], c=colors[labels[i]])

    for i, center in enumerate(centers):
        axs[i].scatter(center[0], center[1], marker='*', s=300, c='r')

        axs[i].set_xlabel('Population Growth (1980-2020)')
        axs[i].set_ylabel('Urban Population Growth (1980-2020)')

    # Add legend to each subplot
    for ax in axs:
        ax.legend(loc='upper left')
    plt.savefig("clustered_data.png")
    plt.show()

# Read in the data
df = pd.read_csv("world_bank_data.csv", skiprows=3, usecols=lambda col: col != 'Unnamed: 66')

# Clean up column names
df.columns = [str(col).strip().replace('\n', '') for col in df.columns]

countries = ['United States', 'China', 'India', 'United Kingdom', 'Canada']
indicators = ["Population growth (annual %)",'Urban population growth (annual %)']
n_clusters = 5

plot_clustered_data(df, countries, indicators, n_clusters)

def plot_country_data(df, countries):
    """
    Plots line graphs of the selected countries' data.

    Parameters:
        df (pandas.DataFrame): the DataFrame containing the data
        countries (list): a list of country names to plot

    Returns:
        None
    """ 
    # Loop over the selected countries
    for country in countries:
        # Extract the desired country
        df_country = df[df["Country Name"] == country]
        df_country = df_country[['Country Name', '1980', '1985','1990','1995', '2000', '2005', '2010', '2015', '2020']]
    
        # Set the index to the country name
        df_country = df_country.set_index('Country Name')
    
        # Convert the data to a numpy array
        data = df_country.to_numpy()
    
        # Create a plot
        fig, ax = plt.subplots(figsize=(10,5))
        ax.bar(df_country.columns, data[0])
    
        # Set the plot title and axis labels
        ax.set_title(country)
        ax.set_xlabel('Year')
        ax.set_ylabel('Percentage')
        plt.savefig(f"{country}.png")

        plt.show()

# Call the modified function with a list of countries
countries = ['United Kingdom', 'India', "China","Canada", "United States"]
plot_country_data(df, countries)


