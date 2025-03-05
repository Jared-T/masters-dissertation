#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Heuristic Flags Implementation for Fuel Transaction Anomaly Detection

This module implements a set of heuristic indicators to identify potentially anomalous
fuel transactions. The implemented flags include:
1. Abnormally large transactions relative to vehicle category, district, and month
2. Suspiciously frequent transactions (less than specified days apart)
3. Fuel price discrepancies compared to official benchmarks

Author: Jared Tavares
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.ticker import FixedLocator, FixedFormatter
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data(filepath):
    """
    Load the transaction dataset from a CSV file.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
        
    Returns:
    --------
    pandas.DataFrame
        Loaded dataset
    """
    try:
        data = pd.read_csv(filepath)
        logger.info(f"Successfully loaded data from {filepath} with {data.shape[0]} rows and {data.shape[1]} columns")
        return data
    except Exception as e:
        logger.error(f"Error loading data from {filepath}: {str(e)}")
        raise

def calculate_transaction_amount_flag(data, multiplier=1.5):
    """
    Flag transactions with amounts significantly higher than the average for the same
    vehicle category, district, and month.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Transaction data
    multiplier : float
        Threshold multiplier for flagging transactions
        
    Returns:
    --------
    pandas.DataFrame
        Data with added transaction amount flag
    """
    logger.info(f"Calculating transaction amount flags with multiplier={multiplier}")
    
    # Calculate the average transaction amount for each vehicle category, district, and month
    data['Average_Category_Amount'] = data.groupby(
        ['RATE CARD CATEGORY', 'District', 'Month Name']
    )['Transaction Amount'].transform('mean')
    
    # Flag transaction amounts that are large for a category
    data['Transaction_Amount_Flag'] = data['Transaction Amount'] > data['Average_Category_Amount'] * multiplier
    
    flag_count = data['Transaction_Amount_Flag'].sum()
    logger.info(f"Identified {flag_count} transactions ({flag_count/len(data)*100:.2f}%) as abnormally large")
    
    return data

def calculate_transaction_frequency_flag(data):
    """
    Flag transactions that occur too frequently (less than 2 days apart) and with an amount
    greater than the average transaction amount for that vehicle category.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Transaction data
        
    Returns:
    --------
    pandas.DataFrame
        Data with added transaction frequency flag
    """
    logger.info("Calculating transaction frequency flags")
    
    # Ensure transaction date is in datetime format
    if not pd.api.types.is_datetime64_dtype(data['Transaction Date']):
        data['Transaction Date'] = pd.to_datetime(data['Transaction Date'])
    
    # Sort data by registration number and transaction date
    data = data.sort_values(by=['REG_NUM', 'Transaction Date'])
    
    # Calculate the difference in days between transactions for each vehicle
    data['Days_Between_Transactions'] = data.groupby('REG_NUM')['Transaction Date'].diff().dt.days
    
    # Flag transactions that occur too frequently and with higher than average amounts
    data['Transaction_Frequency_Flag'] = (
        (data['Days_Between_Transactions'] < 2) & 
        (data['Transaction Amount'] > data['Average_Category_Amount'])
    )
    
    flag_count = data['Transaction_Frequency_Flag'].sum()
    logger.info(f"Identified {flag_count} transactions ({flag_count/len(data)*100:.2f}%) as suspiciously frequent")
    
    return data

def calculate_fuel_price_flag(data, price_threshold=1.0):
    """
    Flag transactions with fuel prices that deviate significantly from the expected prices.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Transaction data
    price_threshold : float
        Threshold for price difference flagging (in rand)
        
    Returns:
    --------
    pandas.DataFrame
        Data with added fuel price flag
    """
    logger.info(f"Calculating fuel price flags with threshold={price_threshold}")
    
    # Official diesel prices and adjustment
    diesel_actual = [22.75, 23.34, 23.43]  # Actual diesel price
    gov_price = 20.64  # Government price
    mean_diesel = sum(diesel_actual) / 3  # Mean diesel price
    diff = mean_diesel - gov_price  # Difference between mean diesel price and government price
    
    # Create adjusted price columns
    data['Coastal Diesel Adjusted'] = data['Coastal Diesel'] + diff
    
    # Calculate price difference based on fuel type
    data['Price Difference'] = data.apply(
        lambda row: abs(row['Coastal Diesel Adjusted'] - row['Estimated Price Per Litre']) 
        if row['Fuel Type'] == 'Diesel' 
        else abs(row['Coastal Petrol'] - row['Estimated Price Per Litre']), 
        axis=1
    )
    
    # Flag transactions with significant price differences
    data['Fuel_Price_Flag'] = data['Price Difference'] > price_threshold
    
    flag_count = data['Fuel_Price_Flag'].sum()
    logger.info(f"Identified {flag_count} transactions ({flag_count/len(data)*100:.2f}%) with suspicious fuel prices")
    
    return data

def calculate_flag_counts(data):
    """
    Calculate the total number of flags for each transaction and categorize them.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Transaction data with individual flags
        
    Returns:
    --------
    pandas.DataFrame
        Data with added flag count column
    """
    logger.info("Calculating total flag counts for each transaction")
    
    # Create a new variable that counts the number of flags for each transaction
    data['Number_of_Flags'] = (
        data['Transaction_Amount_Flag'].astype(int) + 
        data['Transaction_Frequency_Flag'].astype(int) + 
        data['Fuel_Price_Flag'].astype(int)
    )
    
    # Convert to categorical for better memory usage and performance
    data['Number_of_Flags'] = data['Number_of_Flags'].astype('category')
    
    # Log flag distribution
    flag_counts = data['Number_of_Flags'].value_counts().sort_index()
    for flag_count, count in flag_counts.items():
        logger.info(f"Transactions with {flag_count} flags: {count} ({count/len(data)*100:.2f}%)")
    
    return data

def shorten_names(names, max_length=20):
    """
    Shorten long names for plotting purposes.
    
    Parameters:
    -----------
    names : list or pandas.Index
        Names to shorten
    max_length : int
        Maximum length for shortened names
        
    Returns:
    --------
    list
        Shortened names
    """
    shortened_names = []
    for name in names:
        if len(str(name)) > max_length:
            shortened_names.append(str(name)[:max_length] + '...')
        else:
            shortened_names.append(str(name))
    return shortened_names

def countplot(data1, title1, filename):
    """
    Create a count plot for flag counts.
    
    Parameters:
    -----------
    data1 : pandas.Series
        Data for the plot
    title1 : str
        Title for the plot
    filename : str
        Filename for saving the plot
    """
    # Filter the data and calculate the sum of counts for the remaining categories
    filtered_data1 = data1.to_frame()
    
    # Shorten the names for each category
    shortened_names1 = filtered_data1.index.astype(str)
    
    # Create an 8x8 figure
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 8))
    
    # Set the font size for the labels
    label_font_size = 12
    y_label_font_size = 14
    
    # Plot the data for the first subplot
    bars = ax1.bar(shortened_names1, filtered_data1.iloc[:, 0])
    ax1.set_xticks(range(len(shortened_names1)))
    ax1.set_xticklabels(shortened_names1, fontsize=label_font_size)
    ax1.set_ylabel('Count', fontsize=y_label_font_size)
    ax1.set_xlabel('Number of Flags', fontsize=y_label_font_size)
    ax1.set_title(f'a) {title1}')
    
    # Add count values above each bar
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                 str(int(height)), ha='center', va='bottom', fontsize=12)
    
    # Set the y-axis ticks and labels to integer values for both subplots
    yticks1 = ax1.get_yticks().astype(int)
    ax1.yaxis.set_major_locator(FixedLocator(yticks1))
    ax1.yaxis.set_major_formatter(FixedFormatter(yticks1))
    ax1.tick_params(axis='y', labelsize=label_font_size)
    
    # Adjust the spacing
    plt.tight_layout()
    
    # Save the plot as a PDF file
    plt.savefig(f'plots/heuristics/{filename}', format='pdf', bbox_inches='tight')
    
    # Close the plot
    plt.close(fig)

def boxplot_side_by_side_cont(data, cat_var, cont_var1, cont_var2, title1, title2, filename):
    """
    Create side-by-side boxplots for comparing continuous variables across categories.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Data for the plots
    cat_var : str
        Categorical variable for x-axis
    cont_var1 : str
        First continuous variable for y-axis
    cont_var2 : str
        Second continuous variable for y-axis
    title1 : str
        Title for the first plot
    title2 : str
        Title for the second plot
    filename : str
        Filename for saving the plot
    """
    # Setting the aesthetic style of the plots
    sns.set(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))

    # Plot the first boxplot
    sns.boxplot(x=cat_var, y=cont_var1, data=data, ax=ax1, palette="cividis")
    ax1.set_ylabel(cont_var1, fontsize=14)
    ax1.set_xlabel('Number of Flags', fontsize=14)
    ax1.set_title(f'a) {title1}')

    # Plot the second boxplot
    sns.boxplot(x=cat_var, y=cont_var2, data=data, ax=ax2, palette="cividis")
    ax2.set_ylabel(cont_var2, fontsize=14)
    ax2.set_xlabel('Number of Flags', fontsize=14)
    ax2.set_title(f'b) {title2}')

    # Adjust the spacing
    plt.tight_layout()

    # Save the plot as a PDF file with high resolution
    plt.savefig(f'plots/heuristics/{filename}', format='pdf', dpi=300)

    # Close the plot
    plt.close(fig)

def four_stacked_plots(data, categorical_vars, cluster_var, titles, filename, max_categories=8, max_length=20, color_theme='tab10', show_proportions=False):
    """
    Create four stacked bar plots for categorical variables across clusters.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Data for the plots
    categorical_vars : list
        List of categorical variables to plot
    cluster_var : str
        Clustering variable
    titles : list
        List of titles for the plots
    filename : str
        Filename for saving the plots
    max_categories : int
        Maximum number of categories to show
    max_length : int
        Maximum length for category names
    color_theme : str
        Color theme for the plots
    show_proportions : bool
        Whether to show proportion values on the bars
    """
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs = axs.ravel()

    for i, (cat_var, title) in enumerate(zip(categorical_vars, titles)):
        # Calculate the proportions of each category in each cluster
        cluster_proportions = data.groupby([cluster_var, cat_var]).size().unstack(fill_value=0)
        cluster_proportions = cluster_proportions.div(cluster_proportions.sum(axis=1), axis=0)

        # Get the top categories and group the rest into "Others"
        top_categories = cluster_proportions.sum().nlargest(max_categories).index
        cluster_proportions["Others"] = cluster_proportions.drop(columns=top_categories).sum(axis=1)
        cluster_proportions = cluster_proportions[top_categories.tolist() + ["Others"]]

        # Shorten the category names if necessary
        shortened_names = shorten_names(cluster_proportions.columns, max_length=max_length)

        # Check if the cluster labels start from 0 or -1
        if cluster_proportions.index.min() == 0:
            # Shift the cluster labels up by 1 and rename them
            cluster_proportions.index = [str(i+1) for i in cluster_proportions.index]
        elif cluster_proportions.index.min() == -1:
            # Shift the cluster labels up by 1 (excluding -1) and rename them
            cluster_proportions.index = ["None" if i == -1 else str(i+1) for i in cluster_proportions.index]

        # Get the specified color theme
        color_scheme = plt.cm.get_cmap(color_theme, len(cluster_proportions.columns))
        colors = color_scheme(range(len(cluster_proportions.columns)))

        # Create the stacked bar chart in the corresponding subplot
        cluster_proportions.plot(kind='bar', stacked=True, ax=axs[i], legend=False, color=colors)
        axs[i].set_xticklabels(cluster_proportions.index, rotation=0, fontsize=12)
        axs[i].set_xlabel('Cluster', fontsize=14)
        axs[i].set_ylabel('Proportion', fontsize=14)
        axs[i].set_title(f"{chr(97+i)}) {title}")  # Prepend "a) ", "b) ", "c) ", "d) " to the titles

        if show_proportions:
            # Display the actual proportion numbers on the stacked bar plots if the proportion is greater than 0.1
            for j, rect in enumerate(axs[i].patches):
                height = rect.get_height()
                if height > 0.05:
                    axs[i].text(rect.get_x() + rect.get_width() / 2, rect.get_y() + height / 2,
                                f"{height:.2f}", ha='center', va='center', fontsize=9, color='black')

        # Create the legend for each subplot
        axs[i].legend(title='Categories', fontsize=10, labels=shortened_names, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.savefig(f'plots/heuristics/{filename}', format='pdf', bbox_inches='tight', dpi=300)
    plt.close(fig)

def create_proportions_tables(data, categorical_vars, cluster_var, titles, max_categories=8, max_length=20):
    """
    Create tables with proportions of categorical variables across clusters.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Data for the tables
    categorical_vars : list
        List of categorical variables to analyze
    cluster_var : str
        Clustering variable
    titles : list
        List of titles for the tables
    max_categories : int
        Maximum number of categories to include
    max_length : int
        Maximum length for category names
        
    Returns:
    --------
    list
        List of DataFrames with proportion tables
    """
    tables = []

    for cat_var, title in zip(categorical_vars, titles):
        # Calculate the proportions of each category in each cluster
        cluster_proportions = data.groupby([cluster_var, cat_var]).size().unstack(fill_value=0)
        cluster_proportions = cluster_proportions.div(cluster_proportions.sum(axis=1), axis=0)

        # Get the top categories and group the rest into "Others"
        top_categories = cluster_proportions.sum().nlargest(max_categories).index
        cluster_proportions["Others"] = cluster_proportions.drop(columns=top_categories).sum(axis=1)
        cluster_proportions = cluster_proportions[top_categories.tolist() + ["Others"]]

        # Shorten the category names if necessary
        shortened_names = shorten_names(cluster_proportions.columns, max_length=max_length)
        cluster_proportions.columns = shortened_names

        # Check if the cluster labels start from 0 or -1
        if cluster_proportions.index.min() == 0:
            # Shift the cluster labels up by 1 and rename them
            cluster_proportions.index = [str(i+1) for i in cluster_proportions.index]
        elif cluster_proportions.index.min() == -1:
            # Shift the cluster labels up by 1 (excluding -1) and rename them
            cluster_proportions.index = ["None" if i == -1 else str(i+1) for i in cluster_proportions.index]

        # Multiply by 100 and round to 2 decimal places
        cluster_proportions = cluster_proportions.round(2)

        # Add the title as a column to the table
        cluster_proportions.insert(0, 'Cluster', cluster_proportions.index)
        cluster_proportions.index = [title] * len(cluster_proportions)

        # Add the table to the list of tables
        tables.append(cluster_proportions)

    return tables

def plot_flag_distributions(data):
    """
    Create plots showing the distributions of flagged transactions.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Transaction data with flags
    """
    # Create output directory if it doesn't exist
    os.makedirs('plots/heuristics', exist_ok=True)
    
    # Plot of flag counts
    countplot(data['Number_of_Flags'].value_counts(), 'Number of Flags', 'countplot.pdf')
    
    # Filter to remove extreme values
    data_filtered = data[data['Transaction Amount'] < 5000]
    data_filtered = data_filtered[data_filtered['Transaction Amount'] > 0]

    # Create boxplots of transaction amount and number of litres by number of flags
    boxplot_side_by_side_cont(data_filtered, 'Number_of_Flags', 
                          'Transaction Amount', 'No. of Litres', 
                     'Transaction Amount', 'Number of Litres',
                     'boxplots_trans_litres.pdf')
    
    # Create stacked plots of categorical variables by number of flags
    four_stacked_plots(data,
                   ['MODEL DERIVATIVE', 'DEPARTMENT', 'District', 'RATE CARD CATEGORY'],
                   'Number_of_Flags',
                   ['Model Derivative', 'Department', 'District', 'Rate Card Category'],
                   'heuristics_categorical.pdf',
                   max_categories=5, max_length=15, show_proportions=True)
    
    # Create proportion tables
    tables = create_proportions_tables(data,
                        ['MODEL DERIVATIVE', 'DEPARTMENT', 'District', 'RATE CARD CATEGORY'],
                        'Number_of_Flags',
                        ['Model Derivative', 'Department', 'District', 'Rate Card Category'],
                        max_categories=5, max_length=15)
    
    # Save the tables to CSV files
    for i, table in enumerate(tables):
        table.to_csv(f'plots/heuristics/proportions_table_{i}.csv')

def apply_heuristic_flags(data, multiplier=1.5, price_threshold=1.0):
    """
    Apply all heuristic flags to the transaction data.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Transaction data
    multiplier : float
        Threshold multiplier for transaction amount flag
    price_threshold : float
        Threshold for price difference flag
        
    Returns:
    --------
    pandas.DataFrame
        Data with all flags applied
    """
    # Apply each flag sequentially
    data = calculate_transaction_amount_flag(data, multiplier)
    data = calculate_transaction_frequency_flag(data)
    data = calculate_fuel_price_flag(data, price_threshold)
    data = calculate_flag_counts(data)
    
    return data

def main():
    """Main function to execute the heuristic flags analysis."""
    # Load the dataset
    logger.info("Starting heuristic flags analysis")
    data = load_data(os.path.join("data", "Final for clustering.csv"))
    
    # Apply all heuristic flags
    flagged_data = apply_heuristic_flags(data)
    
    # Plot the distributions
    plot_flag_distributions(flagged_data)
    
    # Save the flagged data
    output_path = 'data/Final Transactions With Flags.csv'
    flagged_data.to_csv(output_path, index=False)
    logger.info(f"Saved flagged transactions to {output_path}")
    
    logger.info("Heuristic flags analysis completed successfully")
    
    return flagged_data

if __name__ == "__main__":
    # Execute the main function if the script is run directly
    flagged_data = main()