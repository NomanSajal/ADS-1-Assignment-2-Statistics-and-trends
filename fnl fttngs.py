# -*- coding: utf-8 -*-
"""
Created on Wed May 10 04:44:16 2023

@author: Anika
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def plot_population_growth(df, countries, indicators):
    """
    Plots population growth data for selected countries using polynomial regression.
    
    Parameters:
    df (pandas.DataFrame): the data to analyze
    countries (list): the list of countries to plot
    indicators (list): the list of indicators to plot
    
    Returns:
    None
    """
    
    # Filter the data for the selected countries and indicators
    df = df[df["Country Name"].isin(countries)]
    df = df[df["Indicator Name"].isin(indicators)]
    
    # Filter for the years 2000, 2005, 2010, 2015, and 2020
    df = df[['Country Name','1980', '1985','1990','1995','2000', '2005', '2010', '2015', '2020']]
    
    # Set the index to the country names
    df = df.set_index('Country Name')
    
    # Convert the data to a numpy array
    data = df.to_numpy()
    
    # Define the model function (a polynomial function)
    def func(x, *params):
        return np.polyval(params, x)

    # Define the x data for the model
    xdata = np.array([1980,1985,1990,1995,2000, 2005, 2010, 2015, 2020])

    # Define the x data for the predictions
    xpred = np.array([2030, 2050])
    
    # Loop over each country and fit the model to the data
    for i in range(data.shape[0]):
        # Extract the y data for the current country
        ydata = data[i, :]
        
        # Set an initial guess for the coefficients
        p0 = [0.0, 0.0, 0.0, 0.0, 0.0]

        # Fit the model to the data
        popt, pcov = curve_fit(func, xdata, ydata, p0=p0)

        # Print the optimized coefficients
        print(popt)

        # Make predictions for 10 and 20 years in the future
        ypred = func(xpred, *popt)

        # Calculate the confidence intervals for the predictions
        err = np.sqrt(np.diag(pcov))
        err_ranges = err[:, np.newaxis] * np.array([-1.96, 1.96])[np.newaxis, :]
        conf_intervals = ypred[:, np.newaxis] + err_ranges.T

        # Plot the data, the best-fit curve, and the predicted values
        plt.plot(xdata, ydata, 'bo', label='Data')
        plt.plot(xdata, func(xdata, *popt), 'r-', label='Best Fit')
        plt.plot(np.concatenate((xdata, xpred)), np.concatenate((ydata, ypred)), 'g--', label='Predicted Growth')
        plt.fill_between(xpred, conf_intervals[:,0], conf_intervals[:,1], color='gray', alpha=0.2)
        plt.xlabel('Year')
        plt.ylabel('Population Growth (annual %)')
        plt.title(f'Population Growth in {countries[i]}')
        plt.grid()
        plt.legend()
        plt.show()
        
# Read in the data, excluding the "Unnamed: 66" column
df = pd.read_csv("world_bank_data.csv", skiprows=4, usecols=range(65))

# Define the countries and indicators to plot
countries = ['United States', 'China', 'India', 'United Kingdom', 'Canada']
indicators = ['Population growth (annual %)']

# Plot the population growth data
plot_population_growth(df, countries, indicators)

