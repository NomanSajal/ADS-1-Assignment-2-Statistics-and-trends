# -*- coding: utf-8 -*-
"""
Created on Thu May 11 23:37:26 2023

@author: Anika
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def plot_population_growth():
    """
    Reads in world bank data, extracts desired countries and indicators, filters for years 1980 to 2020,
    performs KMeans clustering, and plots the data and cluster centers.

    Returns:
        None
    """
    # Read in the data
    df = pd.read_csv("world_bank_data.csv", skiprows=3, usecols=lambda col: col != 'Unnamed: 66')

    # Clean up column names
    df.columns = [str(col).strip().replace('\n', '') for col in df.columns]

    # Extract the desired countries and indicators
    countries = ['United States', 'China', 'India', 'United Kingdom', 'Canada']
    indicators = ["Population growth (annual %)"]
    df = df[df["Country Name"].isin(countries)]
    df = df[df["Indicator Name"].isin(indicators)]

    # Filter for the years 1980 to 2020
    df = df[['Country Name', '1980', '1985', '1990', '1995', '2000', '2005', '2010', '2015', '2020']]

    # Set the index to the country names
    df = df.set_index('Country Name')

    # Convert the data to a numpy array
    data = df.to_numpy()

    # Create a KMeans clustering object with 5 clusters
    kmeans = KMeans(n_clusters=5)

    # Fit the KMeans object to the data
    kmeans.fit(data)

    # Get the cluster labels and centers
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    # Set colors for each country
    colors = ['b', 'g', 'r', 'c', 'y']

    # Plot the data and cluster centers
    for i, country in enumerate(countries):
        plt.scatter(data[i, :], data[i, :], c=colors[i], label=country)

    plt.scatter(centers[:, 0], centers[:, 1], marker='*', s=300, c='r')
    plt.scatter(data[:, 0], data[:, -1], c=labels)
    plt.xlabel('1980')
    plt.ylabel('2020')
    plt.title('Population Growth by Country')
    plt.legend()
    plt.show()
    plt.savefig("22014119.png")
plot_population_growth()