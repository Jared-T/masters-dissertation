#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Vehicle Clustering Analysis

This module implements vehicle-level clustering to identify distinct groups of vehicles
based on their fuel transaction patterns. The clustering approaches include:
1. K-means clustering
2. Agglomerative hierarchical clustering

The module also provides visualization tools for analyzing and interpreting the
clustering results.

Author: Jared Tavares
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from matplotlib.ticker import FixedLocator, FixedFormatter
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data(filepath="data/Final KMPL dataset.csv"):
    """
    Load the vehicle dataset from a CSV file.
    
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
        data_agg = pd.read_csv(filepath)
        logger.info(f"Successfully loaded data from {filepath} with {data_agg.shape[0]} rows and {data_agg.shape[1]} columns")
        return data_agg
    except Exception as e:
        logger.error(f"Error loading data from {filepath}: {str(e)}")
        raise

def prepare_data_for_clustering(data_agg):
    """
    Prepare vehicle data for clustering by selecting relevant features,
    handling missing values, and scaling.
    
    Parameters:
    -----------
    data_agg : pandas.DataFrame
        Vehicle dataset
        
    Returns:
    --------
    tuple
        X_scaled (scaled features), data_agg_clusters (DataFrame with vehicle IDs)
    """
    logger.info("Preparing vehicle data for clustering")
    
    # Select the columns for clustering
    columns_for_clustering = ['Reg', 'KMPL', 
                              'Total Transaction Amount', 'Mean Transaction Amount',
                              'Total No. of Litres', 'Mean No. of Litres']
    
    # Extract data for clustering
    data_agg_clustering = data_agg[columns_for_clustering]
    
    # Remove rows with missing values
    data_agg_clusters = data_agg_clustering.dropna()
    logger.info(f"After removing missing values, {data_agg_clusters.shape[0]} vehicles remain")
    
    # Select the data without the vehicle ID for actual clustering
    data_for_clustering = data_agg_clusters.drop('Reg', axis=1)
    
    # Encode the categorical variables if any
    data_agg_encoded = pd.get_dummies(data_for_clustering)
    
    # Convert to numpy array for clustering
    X = data_agg_encoded.values
    
    # Standardize the features
    scaler_agg = StandardScaler()
    X_scaled = scaler_agg.fit_transform(X)
    
    logger.info(f"Data prepared for clustering with {X_scaled.shape[1]} features")
    
    return X_scaled, data_agg_clusters

def find_optimal_clusters_kmeans(X_scaled, max_clusters=10, plot_dir="plots/vehicle_clustering"):
    """
    Find the optimal number of clusters using the elbow method with K-means.
    
    Parameters:
    -----------
    X_scaled : numpy.ndarray
        Scaled features for clustering
    max_clusters : int
        Maximum number of clusters to test
    plot_dir : str
        Directory for saving plots
        
    Returns:
    --------
    int
        Suggested optimal number of clusters
    """
    logger.info(f"Finding optimal number of clusters using elbow method (max={max_clusters})")
    
    # Create output directory if it doesn't exist
    os.makedirs(plot_dir, exist_ok=True)
    
    # Calculate WCSS for different numbers of clusters
    wcss = []
    for i in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=1)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)
    
    # Create a plot
    plt.figure(figsize=(8, 8))
    plt.plot(range(1, max_clusters + 1), wcss, marker='o', linestyle='-', color='darkblue')
    plt.xlabel('Number of Clusters', fontsize=15)
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)', fontsize=15)
    plt.xticks(range(1, max_clusters + 1), fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'{plot_dir}/kmeans_elbow_method_plot.pdf', format='pdf', dpi=300)
    plt.close()
    
    # Simple heuristic to estimate the elbow point
    # Calculate the rate of decrease in WCSS
    wcss_diff = np.diff(wcss)
    wcss_diff_rate = wcss_diff[1:] / wcss_diff[:-1]
    
    # Find the first point where the rate of decrease levels off
    optimal_clusters = 0
    for i, rate in enumerate(wcss_diff_rate, start=3):
        if rate > 0.7:  # Threshold can be adjusted
            optimal_clusters = i
            break
    
    # Default to 3 if heuristic fails
    if optimal_clusters == 0:
        optimal_clusters = 3
        logger.info("Heuristic didn't find clear elbow point, defaulting to 3 clusters")
    else:
        logger.info(f"Heuristic suggests {optimal_clusters} clusters as optimal")
    
    return optimal_clusters

def perform_kmeans_clustering(X_scaled, data_agg_clusters, n_clusters=3):
    """
    Perform K-means clustering on the vehicle data.
    
    Parameters:
    -----------
    X_scaled : numpy.ndarray
        Scaled features for clustering
    data_agg_clusters : pandas.DataFrame
        DataFrame with vehicle IDs
    n_clusters : int
        Number of clusters
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added cluster labels
    """
    logger.info(f"Performing K-means clustering with {n_clusters} clusters")
    
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=1)
    kmeans.fit(X_scaled)
    
    # Add cluster labels to the DataFrame
    data_agg_clusters['KmeansCluster'] = kmeans.labels_
    
    # Log cluster distribution
    cluster_counts = data_agg_clusters['KmeansCluster'].value_counts().sort_index()
    for cluster, count in cluster_counts.items():
        logger.info(f"K-means Cluster {cluster}: {count} vehicles ({count/len(data_agg_clusters)*100:.2f}%)")
    
    return data_agg_clusters

def plot_dendrogram(X_scaled, plot_dir="plots/vehicle_clustering"):
    """
    Compute and plot the dendrogram for hierarchical clustering.
    
    Parameters:
    -----------
    X_scaled : numpy.ndarray
        Scaled features for clustering
    plot_dir : str
        Directory for saving plots
        
    Returns:
    --------
    numpy.ndarray
        Linkage matrix for agglomerative clustering
    """
    logger.info("Computing linkage matrix for dendrogram")
    
    # Create output directory if it doesn't exist
    os.makedirs(plot_dir, exist_ok=True)
    
    # Compute the condensed Euclidean distance matrix
    distance_matrix = pdist(X_scaled, metric='euclidean')
    
    # Perform hierarchical clustering using the Ward method
    Z_agg = linkage(distance_matrix, method='ward')
    
    # Plotting the dendrogram
    plt.figure(figsize=(8, 8))
    dendrogram(Z_agg, no_labels=True, truncate_mode='level', p=5, color_threshold=0) # Labels are omitted for clarity
    plt.xlabel('Sample index', fontsize=15)
    plt.ylabel('Distance', fontsize=15)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    # Save the dendrogram
    plt.savefig(f'{plot_dir}/agglomerative_dendrogram.pdf', format='pdf', dpi=300)
    plt.close()
    
    return Z_agg

def perform_agglomerative_clustering(X_scaled, data_agg_clusters, n_clusters=4):
    """
    Perform agglomerative hierarchical clustering on the vehicle data.
    
    Parameters:
    -----------
    X_scaled : numpy.ndarray
        Scaled features for clustering
    data_agg_clusters : pandas.DataFrame
        DataFrame with vehicle IDs
    n_clusters : int
        Number of clusters
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added cluster labels
    """
    logger.info(f"Performing agglomerative clustering with {n_clusters} clusters")
    
    # Perform Agglomerative Clustering
    agg_clust = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
    clusters = agg_clust.fit_predict(X_scaled)
    
    # Add cluster labels to the DataFrame
    data_agg_clusters['AggCluster'] = clusters
    
    # Log cluster distribution
    cluster_counts = data_agg_clusters['AggCluster'].value_counts().sort_index()
    for cluster, count in cluster_counts.items():
        logger.info(f"Agglomerative Cluster {cluster}: {count} vehicles ({count/len(data_agg_clusters)*100:.2f}%)")
    
    return data_agg_clusters

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

def countplot_side_by_side(data1, data2, title1, title2, filename, plot_dir="plots/vehicle_clustering"):
    """
    Create side-by-side count plots for cluster distributions.
    
    Parameters:
    -----------
    data1, data2 : pandas.Series
        Data for the plots
    title1, title2 : str
        Titles for the plots
    filename : str
        Filename for saving the plot
    plot_dir : str
        Directory for saving plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(plot_dir, exist_ok=True)
    
    # Filter the data and calculate the sum of counts for the remaining categories
    filtered_data1 = data1.to_frame()
    filtered_data2 = data2.to_frame()
    
    # Shorten the names for each category
    shortened_names1 = filtered_data1.index.astype(str)
    shortened_names2 = filtered_data2.index.astype(str)
    
    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
    
    # Set the font size for the labels
    label_font_size = 12
    y_label_font_size = 14
    
    # Plot the data for the first subplot
    ax1.bar(shortened_names1, filtered_data1.iloc[:, 0])
    ax1.set_xticks(range(len(shortened_names1)))
    ax1.set_xticklabels(shortened_names1, fontsize=label_font_size)
    ax1.set_ylabel('Count', fontsize=y_label_font_size)
    ax1.set_xlabel('Cluster', fontsize=y_label_font_size)
    ax1.set_title(f'a) {title1}')
    
    # Plot the data for the second subplot
    ax2.bar(shortened_names2, filtered_data2.iloc[:, 0])
    ax2.set_xticks(range(len(shortened_names2)))
    ax2.set_xticklabels(shortened_names2, fontsize=label_font_size)
    ax2.set_xlabel('Cluster', fontsize=y_label_font_size)
    ax2.set_title(f'b) {title2}')
    
    # Set the y-axis ticks and labels to integer values for both subplots
    yticks1 = ax1.get_yticks().astype(int)
    yticks2 = ax2.get_yticks().astype(int)
    ax1.yaxis.set_major_locator(FixedLocator(yticks1))
    ax1.yaxis.set_major_formatter(FixedFormatter(yticks1))
    ax1.tick_params(axis='y', labelsize=label_font_size)
    ax2.yaxis.set_major_locator(FixedLocator(yticks2))
    ax2.yaxis.set_major_formatter(FixedFormatter(yticks2))
    ax2.tick_params(axis='y', labelsize=label_font_size)
    
    # Adjust the spacing
    plt.tight_layout()
    
    # Save the plot as a PDF file
    plt.savefig(f'{plot_dir}/{filename}', format='pdf', bbox_inches='tight')
    
    # Close the plot
    plt.close(fig)

def boxplot_side_by_side_cat(data, cat_var1, cat_var2, cont_var, title1, title2, filename, plot_dir="plots/vehicle_clustering"):
    """
    Create side-by-side boxplots for comparing continuous variables across clusters.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Data for the plots
    cat_var1, cat_var2 : str
        Categorical variables for x-axis
    cont_var : str
        Continuous variable for y-axis
    title1, title2 : str
        Titles for the plots
    filename : str
        Filename for saving the plot
    plot_dir : str
        Directory for saving plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(plot_dir, exist_ok=True)
    
    # Setting the aesthetic style of the plots
    sns.set(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
    
    # Plot the first boxplot
    sns.boxplot(x=cat_var1, y=cont_var, data=data, ax=ax1, palette="cividis")
    ax1.set_ylabel(cont_var, fontsize=14)
    ax1.set_xlabel('Cluster', fontsize=14)
    ax1.set_title(f'a) {title1}')
    
    # Plot the second boxplot
    sns.boxplot(x=cat_var2, y=cont_var, data=data, ax=ax2, palette="cividis")
    ax2.set_ylabel(cont_var, fontsize=14)
    ax2.set_xlabel('Cluster', fontsize=14)
    ax2.set_title(f'b) {title2}')
    
    # Adjust the spacing
    plt.tight_layout()
    
    # Save the plot as a PDF file with high resolution
    plt.savefig(f'{plot_dir}/{filename}', format='pdf', dpi=300)
    
    # Close the plot
    plt.close(fig)

def four_stacked_plots(data, categorical_vars, cluster_var, titles, filename, 
                     max_categories=8, max_length=20, color_theme='tab10', 
                     show_proportions=False, plot_dir="plots/vehicle_clustering"):
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
    plot_dir : str
        Directory for saving plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(plot_dir, exist_ok=True)
    
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
            # Display the actual proportion numbers on the stacked bar plots if the proportion is greater than specified threshold
            for j, rect in enumerate(axs[i].patches):
                height = rect.get_height()
                if height > 0.05:
                    axs[i].text(rect.get_x() + rect.get_width() / 2, rect.get_y() + height / 2,
                               f"{height:.2f}", ha='center', va='center', fontsize=9, color='black')
        
        # Create the legend for each subplot
        axs[i].legend(title='Categories', fontsize=10, labels=shortened_names, loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/{filename}', format='pdf', bbox_inches='tight', dpi=300)
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

def analyze_cluster_characteristics(data, clusters=['KmeansCluster', 'AggCluster']):
    """
    Analyze and log the characteristics of each cluster.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Data with cluster assignments
    clusters : list
        List of cluster column names
    """
    for cluster_col in clusters:
        logger.info(f"\nAnalyzing {cluster_col} characteristics:")
        
        # Get unique cluster values
        unique_clusters = data[cluster_col].unique()
        
        for cluster_val in sorted(unique_clusters):
            # Filter data for this cluster
            cluster_data = data[data[cluster_col] == cluster_val]
            
            # Calculate summary statistics
            kmpl_mean = cluster_data['KMPL'].mean()
            trans_amount_mean = cluster_data['Mean Transaction Amount'].mean()
            litres_mean = cluster_data['Mean No. of Litres'].mean()
            
            logger.info(f"  Cluster {cluster_val}:")
            logger.info(f"    Count: {len(cluster_data)} vehicles")
            logger.info(f"    Average KMPL: {kmpl_mean:.2f}")
            logger.info(f"    Average Transaction Amount: R{trans_amount_mean:.2f}")
            logger.info(f"    Average Litres per Transaction: {litres_mean:.2f}L")
            
            # Get most common vehicle makes and models in this cluster
            if 'VEHICLE MAKE' in data.columns:
                top_makes = cluster_data['VEHICLE MAKE'].value_counts().head(3)
                logger.info(f"    Top Vehicle Makes: {', '.join([f'{make} ({count})' for make, count in top_makes.items()])}")
            
            if 'MODEL DERIVATIVE' in data.columns:
                top_models = cluster_data['MODEL DERIVATIVE'].value_counts().head(3)
                logger.info(f"    Top Models: {', '.join([f'{model} ({count})' for model, count in top_models.items()])}")
            
            # Get most common departments in this cluster
            if 'DEPARTMENT' in data.columns:
                top_depts = cluster_data['DEPARTMENT'].value_counts().head(3)
                logger.info(f"    Top Departments: {', '.join([f'{dept} ({count})' for dept, count in top_depts.items()])}")

def save_clustered_data(data, filepath="data/final_clusters.csv"):
    """
    Save the clustered data to a CSV file.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Data with cluster assignments
    filepath : str
        Path for saving the CSV file
    """
    try:
        # Increment cluster numbers to start from 1 for readability
        if 'KmeansCluster' in data.columns:
            data['KmeansCluster'] = data['KmeansCluster'] + 1
        
        if 'AggCluster' in data.columns:
            data['AggCluster'] = data['AggCluster'] + 1
        
        # Save to CSV
        data.to_csv(filepath, index=False)
        logger.info(f"Clustered data saved to {filepath}")
    except Exception as e:
        logger.error(f"Error saving clustered data to {filepath}: {str(e)}")

def create_cluster_visualizations(data, plot_dir="plots/vehicle_clustering"):
    """
    Create various visualizations to analyze and understand the clusters.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Data with cluster assignments
    plot_dir : str
        Directory for saving plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(plot_dir, exist_ok=True)
    
    # Add 1 to cluster labels to start from 1 for both methods
    data_for_viz = data.copy()
    
    # Get cluster counts
    kmeans_clust = data_for_viz['KmeansCluster'].value_counts()
    agg_clust = data_for_viz['AggCluster'].value_counts()
    
    # Plot cluster counts
    countplot_side_by_side(kmeans_clust, agg_clust, 
                         'k-means', 'agglomerative', 
                         'cluster_counts.pdf', plot_dir)
    
    # Plot boxplots for transaction amounts by cluster
    boxplot_side_by_side_cat(data_for_viz, 'KmeansCluster', 'AggCluster', 
                           'Mean Transaction Amount', 
                         'K-means Clusters', 'Agglomerative Clusters',
                         'cluster_boxplots_meanstrans.pdf', plot_dir)
    
    # Plot boxplots for litres by cluster
    boxplot_side_by_side_cat(data_for_viz, 'KmeansCluster', 'AggCluster', 
                           'Mean No. of Litres', 
                         'K-means Clusters', 'Agglomerative Clusters',
                         'cluster_boxplots_meanlitres.pdf', plot_dir)
    
    # Plot stacked bar charts for categorical variables by K-means clusters
    if all(col in data.columns for col in ['MODEL DERIVATIVE', 'DEPARTMENT', 'District', 'RATE CARD CATEGORY']):
        four_stacked_plots(data_for_viz,
                         ['MODEL DERIVATIVE', 'DEPARTMENT', 'District', 'RATE CARD CATEGORY'],
                         'KmeansCluster',
                         ['Model Derivative', 'Department', 'District', 'Rate Card Category'],
                         'kmeans_clustered_vehicle_plots.pdf',
                         max_categories=5, max_length=15, show_proportions=True, plot_dir=plot_dir)
        
        # Plot stacked bar charts for categorical variables by agglomerative clusters
        four_stacked_plots(data_for_viz,
                         ['MODEL DERIVATIVE', 'DEPARTMENT', 'District', 'RATE CARD CATEGORY'],
                         'AggCluster',
                         ['Model Derivative', 'Department', 'District', 'Rate Card Category'],
                         'agg_clustered_vehicle_plots.pdf',
                         max_categories=5, max_length=15, show_proportions=True, plot_dir=plot_dir)
        
        # Create proportion tables
        create_proportions_tables(data_for_viz,
                                ['MODEL DERIVATIVE', 'DEPARTMENT', 'District', 'RATE CARD CATEGORY'],
                                'KmeansCluster',
                                ['Model Derivative', 'Department', 'District', 'Rate Card Category'],
                                max_categories=5, max_length=20)
        
        create_proportions_tables(data_for_viz,
                                ['MODEL DERIVATIVE', 'DEPARTMENT', 'District', 'RATE CARD CATEGORY'],
                                'AggCluster', 
                                ['Model Derivative', 'Department', 'District', 'Rate Card Category'],
                                max_categories=5, max_length=20)

def main():
    """Main function to execute vehicle clustering analysis."""
    try:
        logger.info("Starting vehicle clustering analysis")
        
        # Load the dataset
        data_agg = load_data()
        
        # Prepare data for clustering
        X_scaled, data_agg_clusters = prepare_data_for_clustering(data_agg)
        
        # Find optimal number of clusters for K-means
        optimal_k = find_optimal_clusters_kmeans(X_scaled)
        
        # Perform K-means clustering
        data_with_kmeans = perform_kmeans_clustering(X_scaled, data_agg_clusters, n_clusters=optimal_k)
        
        # Plot dendrogram for hierarchical clustering
        Z_agg = plot_dendrogram(X_scaled)
        
        # Perform agglomerative clustering
        data_with_clusters = perform_agglomerative_clustering(X_scaled, data_with_kmeans, n_clusters=optimal_k)
        
        # Analyze cluster characteristics
        analyze_cluster_characteristics(data_with_clusters)
        
        # Create visualizations
        create_cluster_visualizations(data_with_clusters)
        
        # Save clustered data
        save_clustered_data(data_with_clusters)
        
        logger.info("Vehicle clustering analysis completed successfully")
        
        return data_with_clusters
    
    except Exception as e:
        logger.error(f"Error in vehicle clustering analysis: {str(e)}")
        raise

if __name__ == "__main__":
    # Execute the main function if the script is run directly
    clustered_data = main()