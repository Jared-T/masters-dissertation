#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Transaction Clustering Analysis

This module implements transaction-level clustering to identify distinct patterns
in fuel transactions. The clustering approaches include:
1. K-means clustering
2. BIRCH clustering
3. Gaussian Mixture Models (GMM)

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
from sklearn.cluster import KMeans, Birch
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data(filepath="data/Final for clustering.csv"):
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
        trans_df = pd.read_csv(filepath)
        logger.info(f"Successfully loaded data from {filepath} with {trans_df.shape[0]} rows and {trans_df.shape[1]} columns")
        return trans_df
    except Exception as e:
        logger.error(f"Error loading data from {filepath}: {str(e)}")
        raise

def prepare_data_for_clustering(trans_df):
    """
    Prepare transaction data for clustering by selecting relevant features,
    handling missing values, and scaling.
    
    Parameters:
    -----------
    trans_df : pandas.DataFrame
        Transaction dataset
        
    Returns:
    --------
    tuple
        X_scaled (scaled features), data_trans_clustering (DataFrame with selected columns)
    """
    logger.info("Preparing transaction data for clustering")
    
    # Select the columns for clustering
    columns_for_clustering = ['No. of Litres', 'Transaction Amount', 'Fuel Price']
    
    if 'Days Between Transactions' in trans_df.columns:
        columns_for_clustering.append('Days Between Transactions')
    
    # Create fuel price column if it doesn't exist
    if 'Fuel Price' not in trans_df.columns and all(col in trans_df.columns for col in ['Fuel Type', 'Coastal Diesel', 'Coastal Petrol']):
        logger.info("Creating Fuel Price column")
        trans_df['Fuel Price'] = trans_df.apply(
            lambda row: row['Coastal Diesel'] if row['Fuel Type'] == 'Diesel' else row['Coastal Petrol'], 
            axis=1
        )
    
    # Keep only the rows without missing values for clustering
    data_trans_clustering = trans_df[columns_for_clustering].dropna()
    logger.info(f"After removing missing values, {data_trans_clustering.shape[0]} transactions remain")
    
    # Remove outliers (optional)
    for col in columns_for_clustering:
        if col in ['Transaction Amount', 'No. of Litres']:
            q1 = data_trans_clustering[col].quantile(0.01)
            q3 = data_trans_clustering[col].quantile(0.99)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            data_trans_clustering = data_trans_clustering[
                (data_trans_clustering[col] >= lower_bound) & 
                (data_trans_clustering[col] <= upper_bound)
            ]
    
    logger.info(f"After removing outliers, {data_trans_clustering.shape[0]} transactions remain")
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data_trans_clustering)
    
    logger.info(f"Data prepared for clustering with {X_scaled.shape[1]} features")
    
    return X_scaled, data_trans_clustering

def find_optimal_clusters_kmeans(X_scaled, max_clusters=10, plot_dir="plots/transaction_clustering"):
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
    
    # Default to 4 if heuristic fails
    if optimal_clusters == 0:
        optimal_clusters = 4
        logger.info("Heuristic didn't find clear elbow point, defaulting to 4 clusters")
    else:
        logger.info(f"Heuristic suggests {optimal_clusters} clusters as optimal")
    
    return optimal_clusters

def find_optimal_clusters_birch(X_scaled, max_clusters=10, plot_dir="plots/transaction_clustering"):
    """
    Find the optimal number of clusters for BIRCH using silhouette scores.
    
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
        Optimal number of clusters
    """
    logger.info(f"Finding optimal number of clusters for BIRCH using silhouette scores (max={max_clusters})")
    
    # Create output directory if it doesn't exist
    os.makedirs(plot_dir, exist_ok=True)
    
    range_n_clusters = list(range(2, max_clusters + 1))
    silhouette_scores_list = []
    
    for n_clusters in range_n_clusters:
        birch = Birch(n_clusters=n_clusters)
        cluster_labels = birch.fit_predict(X_scaled)
        silhouette_avg = silhouette_score(X_scaled, cluster_labels)
        silhouette_scores_list.append(silhouette_avg)
        logger.info(f"Silhouette Score for {n_clusters} clusters: {silhouette_avg:.4f}")
    
    # Plotting silhouette scores with a more professional aesthetic
    plt.figure(figsize=(8, 8))
    plt.plot(range_n_clusters, silhouette_scores_list, marker='o', linestyle='-', color='darkblue')
    plt.xlabel('Number of clusters', fontsize=15)
    plt.ylabel('Silhouette Score', fontsize=15)
    plt.xticks(range_n_clusters, fontsize=12)
    plt.yticks(np.round(np.linspace(min(silhouette_scores_list), max(silhouette_scores_list), 5), 2), fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    # Save the plot as a PDF file with high resolution
    plt.savefig(f'{plot_dir}/silhouette_scores_birch.pdf', format='pdf', dpi=300)
    plt.close()
    
    # Find the optimal number of clusters (highest silhouette score)
    optimal_clusters = range_n_clusters[np.argmax(silhouette_scores_list)]
    logger.info(f"Optimal number of clusters for BIRCH based on silhouette score: {optimal_clusters}")
    
    return optimal_clusters

def perform_kmeans_clustering(X_scaled, data_trans_clustering, n_clusters=4, trans_df=None):
    """
    Perform K-means clustering on the transaction data.
    
    Parameters:
    -----------
    X_scaled : numpy.ndarray
        Scaled features for clustering
    data_trans_clustering : pandas.DataFrame
        DataFrame with selected columns for clustering
    n_clusters : int
        Number of clusters
    trans_df : pandas.DataFrame, optional
        Original transaction DataFrame to add cluster labels
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added cluster labels (original if provided, otherwise clustering DataFrame)
    """
    logger.info(f"Performing K-means clustering with {n_clusters} clusters")
    
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=1)
    kmeans_labels = kmeans.fit_predict(X_scaled)
    
    # Add cluster labels to the DataFrame
    if trans_df is not None:
        # Create a Series for cluster labels with the same index as data_trans_clustering
        cluster_labels = pd.Series(kmeans_labels, index=data_trans_clustering.index)
        
        # Add column to original DataFrame
        if 'TransKmeansCluster' in trans_df.columns:
            trans_df = trans_df.drop('TransKmeansCluster', axis=1)
        
        trans_df = trans_df.join(cluster_labels.rename('TransKmeansCluster'), how='left')
        result_df = trans_df
    else:
        # Add directly to clustering DataFrame
        data_trans_clustering['KmeansCluster'] = kmeans_labels
        result_df = data_trans_clustering
    
    # Log cluster distribution
    cluster_counts = pd.Series(kmeans_labels).value_counts().sort_index()
    for cluster, count in cluster_counts.items():
        logger.info(f"K-means Cluster {cluster}: {count} transactions ({count/len(kmeans_labels)*100:.2f}%)")
    
    return result_df

def perform_birch_clustering(X_scaled, data_trans_clustering, n_clusters=3, trans_df=None):
    """
    Perform BIRCH clustering on the transaction data.
    
    Parameters:
    -----------
    X_scaled : numpy.ndarray
        Scaled features for clustering
    data_trans_clustering : pandas.DataFrame
        DataFrame with selected columns for clustering
    n_clusters : int
        Number of clusters
    trans_df : pandas.DataFrame, optional
        Original transaction DataFrame to add cluster labels
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added cluster labels (original if provided, otherwise clustering DataFrame)
    """
    logger.info(f"Performing BIRCH clustering with {n_clusters} clusters")
    
    # Apply BIRCH clustering
    birch = Birch(n_clusters=n_clusters)
    birch_labels = birch.fit_predict(X_scaled)
    
    # Add cluster labels to the DataFrame
    if trans_df is not None:
        # Create a Series for cluster labels with the same index as data_trans_clustering
        cluster_labels = pd.Series(birch_labels, index=data_trans_clustering.index)
        
        # Add column to original DataFrame
        if 'TransBirchCluster' in trans_df.columns:
            trans_df = trans_df.drop('TransBirchCluster', axis=1)
        
        trans_df = trans_df.join(cluster_labels.rename('TransBirchCluster'), how='left')
        result_df = trans_df
    else:
        # Add directly to clustering DataFrame
        data_trans_clustering['BirchCluster'] = birch_labels
        result_df = data_trans_clustering
    
    # Log cluster distribution
    cluster_counts = pd.Series(birch_labels).value_counts().sort_index()
    for cluster, count in cluster_counts.items():
        logger.info(f"BIRCH Cluster {cluster}: {count} transactions ({count/len(birch_labels)*100:.2f}%)")
    
    return result_df

def perform_gmm_clustering(X_scaled, data_trans_clustering, n_components=4, trans_df=None):
    """
    Perform Gaussian Mixture Model clustering on the transaction data.
    
    Parameters:
    -----------
    X_scaled : numpy.ndarray
        Scaled features for clustering
    data_trans_clustering : pandas.DataFrame
        DataFrame with selected columns for clustering
    n_components : int
        Number of mixture components
    trans_df : pandas.DataFrame, optional
        Original transaction DataFrame to add cluster labels
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added cluster labels (original if provided, otherwise clustering DataFrame)
    """
    logger.info(f"Performing GMM clustering with {n_components} components")
    
    # Ensure X_scaled is a dense matrix
    if hasattr(X_scaled, 'toarray'):
        X_dense = X_scaled.toarray()
    else:
        X_dense = X_scaled
    
    # Apply Gaussian Mixture Model
    gmm = GaussianMixture(n_components=n_components, random_state=1)
    gmm_labels = gmm.fit_predict(X_dense)
    
    # Add cluster labels to the DataFrame
    if trans_df is not None:
        # Create a Series for cluster labels with the same index as data_trans_clustering
        cluster_labels = pd.Series(gmm_labels, index=data_trans_clustering.index)
        
        # Add column to original DataFrame
        if 'TransGMMCluster' in trans_df.columns:
            trans_df = trans_df.drop('TransGMMCluster', axis=1)
        
        trans_df = trans_df.join(cluster_labels.rename('TransGMMCluster'), how='left')
        result_df = trans_df
    else:
        # Add directly to clustering DataFrame
        data_trans_clustering['GMMCluster'] = gmm_labels
        result_df = data_trans_clustering
    
    # Log cluster distribution
    cluster_counts = pd.Series(gmm_labels).value_counts().sort_index()
    for cluster, count in cluster_counts.items():
        logger.info(f"GMM Cluster {cluster}: {count} transactions ({count/len(gmm_labels)*100:.2f}%)")
    
    return result_df

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

def boxplot_2x2_cat(data, cat_var1, cat_var2, cat_var3, cont_var, title1, title2, title3, filename, plot_dir="plots/transaction_clustering"):
    """
    Create a 2x2 grid of boxplots for comparing a continuous variable across different clustering methods.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Data for the plots
    cat_var1, cat_var2, cat_var3 : str
        Categorical variables for x-axis (cluster columns)
    cont_var : str
        Continuous variable for y-axis
    title1, title2, title3 : str
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
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))
    
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
    
    # Plot the third boxplot
    sns.boxplot(x=cat_var3, y=cont_var, data=data, ax=ax3, palette="cividis")
    ax3.set_ylabel(cont_var, fontsize=14)
    ax3.set_xlabel('Cluster', fontsize=14)
    ax3.set_title(f'c) {title3}')
    
    # Remove the fourth subplot (unused)
    ax4.axis('off')
    
    # Adjust the spacing
    plt.tight_layout()
    
    # Save the plot as a PDF file with high resolution
    plt.savefig(f'{plot_dir}/{filename}', format='pdf', dpi=300)
    
    # Close the plot
    plt.close(fig)

def plot_categorical_distributions(data, cat_dist_var, cluster_vars, filename, plot_dir="plots/transaction_clustering"):
    """
    Create a plot showing the distribution of a categorical variable across different clusters.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Data for the plots
    cat_dist_var : str
        Categorical variable to analyze distribution
    cluster_vars : list
        List of clustering column names
    filename : str
        Filename for saving the plot
    plot_dir : str
        Directory for saving plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(plot_dir, exist_ok=True)
    
    # Process categorical variable to include top categories and 'Other'
    top_categories = data[cat_dist_var].value_counts().nlargest(5).index
    data_plot = data.copy()
    data_plot[cat_dist_var] = data_plot[cat_dist_var].apply(lambda x: x if x in top_categories else 'Other')
    
    # Determine the grid layout
    if len(cluster_vars) == 3:
        nrows, ncols = 2, 2
        fig_size = (10, 10)
    else:
        nrows, ncols = 1, 2
        fig_size = (12, 7)
    
    fig, axs = plt.subplots(nrows, ncols, figsize=fig_size, dpi=150)
    
    # Adjust axs to be a 2D array for consistency
    if nrows * ncols == 2:
        axs = axs.reshape(1, -1)
    
    plot_prefix = ['a)', 'b)', 'c)']
    
    for i, variable in enumerate(cluster_vars):
        ax = axs[i // ncols, i % ncols]
        sns.countplot(x=variable, hue=cat_dist_var, data=data_plot, ax=ax, alpha=0.7)
        ax.set_title(f'{plot_prefix[i]} {cat_dist_var} by {variable}', fontsize=16)
        ax.set_xlabel(variable, fontsize=15)
        ax.set_ylabel('Count', fontsize=15)
        ax.tick_params(axis='x', labelrotation=45)
        ax.legend(title=cat_dist_var, loc='upper right')
    
    # Hide the unused subplot if there is one
    if len(cluster_vars) < nrows * ncols:
        for i in range(len(cluster_vars), nrows * ncols):
            axs[i // ncols, i % ncols].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/{filename}', format='pdf', bbox_inches='tight')
    plt.close()

def analyze_cluster_characteristics(data, cluster_columns=['TransKmeansCluster', 'TransBirchCluster', 'TransGMMCluster']):
    """
    Analyze and log the characteristics of each cluster for each clustering method.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Data with cluster assignments
    cluster_columns : list
        List of cluster column names
    """
    for cluster_col in cluster_columns:
        if cluster_col not in data.columns:
            logger.warning(f"Cluster column {cluster_col} not found in data")
            continue
        
        logger.info(f"\nAnalyzing {cluster_col} characteristics:")
        
        # Get unique cluster values
        unique_clusters = data[cluster_col].dropna().unique()
        
        for cluster_val in sorted(unique_clusters):
            # Filter data for this cluster
            cluster_data = data[data[cluster_col] == cluster_val]
            
            # Calculate summary statistics
            trans_amount_mean = cluster_data['Transaction Amount'].mean()
            litres_mean = cluster_data['No. of Litres'].mean()
            
            logger.info(f"  Cluster {cluster_val}:")
            logger.info(f"    Count: {len(cluster_data)} transactions")
            logger.info(f"    Average Transaction Amount: R{trans_amount_mean:.2f}")
            logger.info(f"    Average Litres: {litres_mean:.2f}L")
            
            # Get most common departments in this cluster
            if 'DEPARTMENT' in data.columns:
                top_depts = cluster_data['DEPARTMENT'].value_counts().head(3)
                logger.info(f"    Top Departments: {', '.join([f'{dept} ({count})' for dept, count in top_depts.items()])}")
            
            # Get most common vehicle makes and models in this cluster
            if 'VEHICLE MAKE' in data.columns:
                top_makes = cluster_data['VEHICLE MAKE'].value_counts().head(3)
                logger.info(f"    Top Vehicle Makes: {', '.join([f'{make} ({count})' for make, count in top_makes.items()])}")
            
            if 'MODEL DERIVATIVE' in data.columns:
                top_models = cluster_data['MODEL DERIVATIVE'].value_counts().head(3)
                logger.info(f"    Top Models: {', '.join([f'{model} ({count})' for model, count in top_models.items()])}")
            
            # Get most common districts in this cluster
            if 'District' in data.columns:
                top_districts = cluster_data['District'].value_counts().head(3)
                logger.info(f"    Top Districts: {', '.join([f'{district} ({count})' for district, count in top_districts.items()])}")

def prepare_clusters_for_visualization(data, cluster_columns=['TransKmeansCluster', 'TransBirchCluster', 'TransGMMCluster']):
    """
    Prepare cluster labels for visualization by converting to user-friendly format.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Data with cluster assignments
    cluster_columns : list
        List of cluster column names
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with formatted cluster labels
    """
    # Make a copy to avoid modifying the original
    formatted_data = data.copy()
    
    # Rename columns for better readability
    column_mapping = {
        'TransKmeansCluster': 'KMeans Cluster',
        'TransBirchCluster': 'Birch Cluster',
        'TransGMMCluster': 'GMM Cluster'
    }
    
    for old_col, new_col in column_mapping.items():
        if old_col in formatted_data.columns:
            formatted_data = formatted_data.rename(columns={old_col: new_col})
            
            # Update cluster_columns list
            index = cluster_columns.index(old_col)
            cluster_columns[index] = new_col
    
    # Drop rows with NaN in all cluster columns
    formatted_data = formatted_data.dropna(subset=cluster_columns, how='all')
    
    # Adjust each cluster column
    for column in cluster_columns:
        if column in formatted_data.columns:
            # Increment cluster number by 1 if they start at 0
            if formatted_data[column].min() <= 0.0:
                formatted_data[column] = formatted_data[column] + 1
            
            # Convert to string and prepend with "Cluster "
            formatted_data[column] = 'Cluster ' + formatted_data[column].astype(str)
            
            # Remove any trailing '.0' from the string representation
            formatted_data[column] = formatted_data[column].replace(to_replace=r'\.0$', value='', regex=True)
    
    return formatted_data, cluster_columns

def create_cluster_visualizations(data, cluster_columns, plot_dir="plots/transaction_clustering"):
    """
    Create various visualizations to analyze and understand the transaction clusters.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Data with cluster assignments
    cluster_columns : list
        List of cluster column names
    plot_dir : str
        Directory for saving plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(plot_dir, exist_ok=True)
    
    # Create boxplots for transaction amount by cluster
    boxplot_2x2_cat(
        data, 
        cluster_columns[0], cluster_columns[1], cluster_columns[2],
        'Transaction Amount', 
        'K-means', 'BIRCH', 'GMM',
        'transaction_amount_boxplots.pdf', 
        plot_dir
    )
    
    # Create boxplots for litres by cluster
    boxplot_2x2_cat(
        data, 
        cluster_columns[0], cluster_columns[1], cluster_columns[2],
        'No. of Litres', 
        'K-means', 'BIRCH', 'GMM',
        'litres_boxplots.pdf', 
        plot_dir
    )
    
    # Create categorical distribution plots
    if 'District' in data.columns:
        plot_categorical_distributions(
            data, 
            'District', 
            cluster_columns,
            'district_distribution.pdf', 
            plot_dir
        )
    
    if 'VEHICLE MAKE' in data.columns:
        plot_categorical_distributions(
            data, 
            'VEHICLE MAKE', 
            cluster_columns,
            'make_distribution.pdf', 
            plot_dir
        )
    
    if 'MODEL DERIVATIVE' in data.columns:
        plot_categorical_distributions(
            data, 
            'MODEL DERIVATIVE', 
            cluster_columns,
            'model_distribution.pdf', 
            plot_dir
        )

def save_clustered_data(data, filepath="data/final_transactions_with_clusters.csv"):
    """
    Save the clustered transaction data to a CSV file.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Data with cluster assignments
    filepath : str
        Path for saving the CSV file
    """
    try:
        # Save to CSV
        data.to_csv(filepath, index=False)
        logger.info(f"Clustered transaction data saved to {filepath}")
    except Exception as e:
        logger.error(f"Error saving clustered data to {filepath}: {str(e)}")

def main():
    """Main function to execute transaction clustering analysis."""
    try:
        logger.info("Starting transaction clustering analysis")
        
        # Load the dataset
        trans_df = load_data()
        
        # Prepare data for clustering
        X_scaled, data_trans_clustering = prepare_data_for_clustering(trans_df)
        
        # Find optimal number of clusters for K-means
        optimal_k = find_optimal_clusters_kmeans(X_scaled)
        
        # Find optimal number of clusters for BIRCH
        optimal_birch = find_optimal_clusters_birch(X_scaled)
        
        # Perform clustering with the optimal number of clusters
        trans_df_kmeans = perform_kmeans_clustering(
            X_scaled, data_trans_clustering, n_clusters=optimal_k, trans_df=trans_df
        )
        
        trans_df_birch = perform_birch_clustering(
            X_scaled, data_trans_clustering, n_clusters=optimal_birch, trans_df=trans_df_kmeans
        )
        
        trans_df_final = perform_gmm_clustering(
            X_scaled, data_trans_clustering, n_components=optimal_k, trans_df=trans_df_birch
        )
        
        # Analyze cluster characteristics
        analyze_cluster_characteristics(trans_df_final)
        
        # Prepare clusters for visualization
        vis_data, vis_cluster_columns = prepare_clusters_for_visualization(trans_df_final)
        
        # Create visualizations
        create_cluster_visualizations(vis_data, vis_cluster_columns)
        
        # Save clustered data
        save_clustered_data(trans_df_final)
        
        logger.info("Transaction clustering analysis completed successfully")
        
        return trans_df_final
    
    except Exception as e:
        logger.error(f"Error in transaction clustering analysis: {str(e)}")
        raise

if __name__ == "__main__":
    # Execute the main function if the script is run directly
    clustered_data = main()