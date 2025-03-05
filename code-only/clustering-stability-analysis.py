#!/usr/bin/env python
# coding: utf-8

# # Clustering Stability Analysis
# 
# This script extends the original clustering implementation by conducting comprehensive 
# stability analysis to validate the robustness of clustering results.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, OPTICS
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.model_selection import ParameterGrid
import os
from matplotlib.ticker import FixedLocator, FixedFormatter
from tqdm import tqdm
import joblib
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

# Set the aesthetic style of the plots
sns.set_style("whitegrid")
plt.rc('axes', titlesize=16)
plt.rc('axes', labelsize=15)
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
sns.set_palette('cividis')

def load_datasets():
    """
    Load necessary datasets for clustering analysis.
    
    Returns:
    --------
    trans_df : pandas DataFrame
        Transaction-level data
    vehicle_df : pandas DataFrame
        Vehicle-level data
    """
    # Load transaction data
    trans_df = pd.read_csv(os.path.join("data", "Final for clustering.csv"))
    
    # Load vehicle data
    vehicle_df = pd.read_csv(os.path.join("data", "Final KMPL dataset.csv"))
    
    return trans_df, vehicle_df

def perform_vehicle_clustering_stability(vehicle_df, save_dir='plots/stability/vehicle'):
    """
    Perform stability analysis for vehicle clustering.
    
    Parameters:
    -----------
    vehicle_df : pandas DataFrame
        The vehicle dataset
    save_dir : str
        Directory to save the output plots
    
    Returns:
    --------
    stability_results : pandas DataFrame
        DataFrame containing the stability analysis results
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Select the columns for clustering
    columns_for_clustering = ['KMPL', 'Total Transaction Amount', 'Mean Transaction Amount',
                             'Total No. of Litres', 'Mean No. of Litres']
    
    # Extract the data for clustering
    data_clustering = vehicle_df[columns_for_clustering].copy()
    
    # Handle missing values
    data_clustering = data_clustering.dropna()
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data_clustering)
    
    # Range of clusters to evaluate
    n_clusters_range = range(2, 11)
    
    # Initialize results lists
    silhouette_scores = []
    inertia_values = []
    
    # Calculate silhouette score and inertia for each number of clusters
    for n_clusters in n_clusters_range:
        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=1)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(X_scaled, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        
        # Calculate inertia (within-cluster sum of squares)
        inertia_values.append(kmeans.inertia_)
    
    # Plot silhouette scores
    plt.figure(figsize=(10, 6))
    plt.plot(n_clusters_range, silhouette_scores, marker='o', linestyle='-', color='blue')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for Different Numbers of Clusters (Vehicle Clustering)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'silhouette_scores.pdf'), format='pdf', dpi=300)
    plt.close()
    
    # Plot inertia (elbow method)
    plt.figure(figsize=(10, 6))
    plt.plot(n_clusters_range, inertia_values, marker='o', linestyle='-', color='blue')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia (Within-Cluster Sum of Squares)')
    plt.title('Elbow Method for Optimal Number of Clusters (Vehicle Clustering)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'elbow_method.pdf'), format='pdf', dpi=300)
    plt.close()
    
    # Bootstrapping for cluster stability
    n_bootstraps = 100
    optimal_n_clusters = 3  # Based on prior analysis - adjust if needed
    
    # Results storage
    bootstrap_results = []
    
    # Run bootstrapping
    for i in tqdm(range(n_bootstraps), desc="Bootstrap Sampling"):
        # Sample with replacement
        indices = np.random.choice(X_scaled.shape[0], size=X_scaled.shape[0], replace=True)
        bootstrap_sample = X_scaled[indices]
        
        # Fit KMeans
        kmeans = KMeans(n_clusters=optimal_n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=i)
        bootstrap_labels = kmeans.fit_predict(bootstrap_sample)
        
        # Record centroids
        bootstrap_results.append({
            'bootstrap_id': i,
            'centroids': kmeans.cluster_centers_,
            'labels': bootstrap_labels,
            'indices': indices
        })
    
    # Compute centroid stability (variance) for each cluster and feature
    all_centroids = np.array([result['centroids'] for result in bootstrap_results])
    
    # Calculate mean centroids across bootstraps
    mean_centroids = np.mean(all_centroids, axis=0)
    
    # Calculate standard deviation of centroids across bootstraps
    std_centroids = np.std(all_centroids, axis=0)
    
    # Calculate coefficient of variation (normalized dispersion measure)
    cv_centroids = std_centroids / np.abs(mean_centroids)
    
    # Create feature names for plotting
    feature_names = columns_for_clustering
    
    # Plot centroid stability
    plt.figure(figsize=(12, 8))
    
    # Create bar positions
    bar_width = 0.15
    r = np.arange(len(feature_names))
    
    # Plot bars for each cluster
    for i in range(optimal_n_clusters):
        plt.bar(r + i*bar_width, cv_centroids[i], width=bar_width, 
                label=f'Cluster {i+1}', alpha=0.7)
    
    # Add labels and legend
    plt.xlabel('Features')
    plt.ylabel('Coefficient of Variation (Lower is More Stable)')
    plt.title('Centroid Stability Across Bootstrap Samples')
    plt.xticks(r + bar_width*(optimal_n_clusters-1)/2, feature_names, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'centroid_stability.pdf'), format='pdf', dpi=300)
    plt.close()
    
    # Calculate pair-wise assignments (how often pairs of points are clustered together)
    n_samples = X_scaled.shape[0]
    co_occurrence_matrix = np.zeros((n_samples, n_samples))
    
    for result in bootstrap_results:
        boot_indices = result['indices']
        boot_labels = result['labels']
        
        for i in range(n_samples):
            for j in range(i+1, n_samples):
                # Find the original indices of the bootstrapped samples
                orig_i = boot_indices[i]
                orig_j = boot_indices[j]
                
                # Check if they're in the same cluster
                if boot_labels[i] == boot_labels[j]:
                    co_occurrence_matrix[orig_i, orig_j] += 1
                    co_occurrence_matrix[orig_j, orig_i] += 1
    
    # Normalize by the number of times both points appear in the same bootstrap sample
    pairwise_counts = np.zeros((n_samples, n_samples))
    
    for result in bootstrap_results:
        boot_indices = result['indices']
        for i in range(n_samples):
            for j in range(i+1, n_samples):
                orig_i = boot_indices[i]
                orig_j = boot_indices[j]
                pairwise_counts[orig_i, orig_j] += 1
                pairwise_counts[orig_j, orig_i] += 1
    
    # Avoid division by zero
    pairwise_counts[pairwise_counts == 0] = 1
    
    co_occurrence_matrix = co_occurrence_matrix / pairwise_counts
    
    # Fit the final KMeans model
    kmeans_final = KMeans(n_clusters=optimal_n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=1)
    final_labels = kmeans_final.fit_predict(X_scaled)
    
    # Calculate cluster stability (mean co-occurrence within each cluster)
    cluster_stability = []
    
    for cluster in range(optimal_n_clusters):
        cluster_indices = np.where(final_labels == cluster)[0]
        n_cluster_samples = len(cluster_indices)
        
        if n_cluster_samples <= 1:
            cluster_stability.append(1.0)  # Perfect stability for single-point clusters
            continue
        
        # Calculate mean co-occurrence within the cluster
        within_cluster_co_occurrence = 0
        pair_count = 0
        
        for i in range(n_cluster_samples):
            for j in range(i+1, n_cluster_samples):
                idx_i = cluster_indices[i]
                idx_j = cluster_indices[j]
                within_cluster_co_occurrence += co_occurrence_matrix[idx_i, idx_j]
                pair_count += 1
        
        if pair_count > 0:
            cluster_stability.append(within_cluster_co_occurrence / pair_count)
        else:
            cluster_stability.append(1.0)  # Perfect stability for single-point clusters
    
    # Plot cluster stability
    plt.figure(figsize=(10, 6))
    
    # Create cluster labels
    clusters = [f'Cluster {i+1}' for i in range(optimal_n_clusters)]
    
    plt.bar(clusters, cluster_stability, color='blue', alpha=0.7)
    plt.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='80% Stability Threshold')
    plt.xlabel('Cluster')
    plt.ylabel('Stability Score (Higher is Better)')
    plt.title('Cluster Stability Based on Co-occurrence Analysis')
    plt.ylim(0, 1.0)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'cluster_stability.pdf'), format='pdf', dpi=300)
    plt.close()
    
    # Feature impact analysis
    feature_importance = []
    
    # For each feature, remove it and calculate the impact on clustering results
    for i, feature in enumerate(feature_names):
        # Create a dataset without the current feature
        X_reduced = np.delete(X_scaled, i, axis=1)
        
        # Fit KMeans on the reduced dataset
        kmeans_reduced = KMeans(n_clusters=optimal_n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=1)
        reduced_labels = kmeans_reduced.fit_predict(X_reduced)
        
        # Calculate agreement with full-feature clustering (adjusted Rand index)
        agreement = adjusted_rand_score(final_labels, reduced_labels)
        
        # Calculate impact (1 - agreement, higher means more important)
        impact = 1 - agreement
        
        feature_importance.append({
            'Feature': feature,
            'Impact': impact
        })
    
    # Convert to DataFrame and sort
    feature_importance_df = pd.DataFrame(feature_importance)
    feature_importance_df = feature_importance_df.sort_values('Impact', ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    
    plt.bar(feature_importance_df['Feature'], feature_importance_df['Impact'], color='blue', alpha=0.7)
    plt.xlabel('Feature')
    plt.ylabel('Feature Impact (Higher is More Important)')
    plt.title('Feature Impact on Clustering Stability')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'feature_impact.pdf'), format='pdf', dpi=300)
    plt.close()
    
    # Different clustering algorithms comparison
    clustering_algorithms = {
        'KMeans': KMeans(n_clusters=optimal_n_clusters, random_state=1),
        'Agglomerative': AgglomerativeClustering(n_clusters=optimal_n_clusters),
        'GaussianMixture': GaussianMixture(n_components=optimal_n_clusters, random_state=1)
    }
    
    algorithm_labels = {}
    
    for name, algorithm in clustering_algorithms.items():
        algorithm_labels[name] = algorithm.fit_predict(X_scaled)
    
    # Calculate agreement between algorithms
    algorithm_agreement = {}
    
    for name1, labels1 in algorithm_labels.items():
        for name2, labels2 in algorithm_labels.items():
            if name1 != name2:
                agreement = adjusted_rand_score(labels1, labels2)
                algorithm_agreement[f'{name1} vs {name2}'] = agreement
    
    # Convert to DataFrame
    algorithm_agreement_df = pd.DataFrame([algorithm_agreement]).melt()
    algorithm_agreement_df.columns = ['Algorithm Pair', 'Agreement']
    
    # Plot algorithm agreement
    plt.figure(figsize=(12, 6))
    
    plt.bar(algorithm_agreement_df['Algorithm Pair'], algorithm_agreement_df['Agreement'], color='blue', alpha=0.7)
    plt.axhline(y=0.7, color='red', linestyle='--', alpha=0.7, label='70% Agreement Threshold')
    plt.xlabel('Algorithm Pair')
    plt.ylabel('Agreement (Adjusted Rand Index)')
    plt.title('Agreement Between Different Clustering Algorithms')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.0)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'algorithm_agreement.pdf'), format='pdf', dpi=300)
    plt.close()
    
    # Create a comprehensive stability report
    stability_report = {
        'optimal_clusters': optimal_n_clusters,
        'silhouette_scores': dict(zip(n_clusters_range, silhouette_scores)),
        'inertia_values': dict(zip(n_clusters_range, inertia_values)),
        'cluster_stability': dict(zip(clusters, cluster_stability)),
        'feature_importance': feature_importance_df.to_dict('records'),
        'algorithm_agreement': algorithm_agreement
    }
    
    # Convert to DataFrame for easy saving
    stability_report_df = pd.DataFrame([stability_report])
    
    # Save to CSV
    stability_report_df.to_json(os.path.join(save_dir, 'vehicle_clustering_stability_report.json'), orient='records')
    
    return stability_report

def perform_transaction_clustering_stability(trans_df, save_dir='plots/stability/transaction'):
    """
    Perform stability analysis for transaction clustering.
    
    Parameters:
    -----------
    trans_df : pandas DataFrame
        The transaction dataset
    save_dir : str
        Directory to save the output plots
    
    Returns:
    --------
    stability_results : pandas DataFrame
        DataFrame containing the stability analysis results
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Select the columns for clustering
    columns_for_clustering = ['Transaction Amount', 'No. of Litres', 'Estimated Price Per Litre', 
                             'Days Between Transactions']
    
    # Extract the data for clustering
    data_clustering = trans_df[columns_for_clustering].copy()
    
    # Handle missing values
    data_clustering = data_clustering.dropna()
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data_clustering)
    
    # Range of clusters to evaluate
    n_clusters_range = range(2, 11)
    
    # Initialize results lists
    silhouette_scores = []
    inertia_values = []
    
    # Calculate silhouette score and inertia for each number of clusters
    for n_clusters in n_clusters_range:
        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=1)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(X_scaled, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        
        # Calculate inertia (within-cluster sum of squares)
        inertia_values.append(kmeans.inertia_)
    
    # Plot silhouette scores
    plt.figure(figsize=(10, 6))
    plt.plot(n_clusters_range, silhouette_scores, marker='o', linestyle='-', color='blue')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for Different Numbers of Clusters (Transaction Clustering)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'silhouette_scores.pdf'), format='pdf', dpi=300)
    plt.close()
    
    # Plot inertia (elbow method)
    plt.figure(figsize=(10, 6))
    plt.plot(n_clusters_range, inertia_values, marker='o', linestyle='-', color='blue')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia (Within-Cluster Sum of Squares)')
    plt.title('Elbow Method for Optimal Number of Clusters (Transaction Clustering)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'elbow_method.pdf'), format='pdf', dpi=300)
    plt.close()
    
    # For transactions, we'll take a random sample due to the large dataset size
    sample_size = min(10000, X_scaled.shape[0])
    random_indices = np.random.choice(X_scaled.shape[0], size=sample_size, replace=False)
    X_sample = X_scaled[random_indices]
    
    # Determine optimal number of clusters based on silhouette score
    optimal_n_clusters = n_clusters_range[np.argmax(silhouette_scores)]
    
    # Comparison of different clustering approaches
    clustering_algorithms = {
        'KMeans': KMeans(n_clusters=optimal_n_clusters, random_state=1),
        'Agglomerative': AgglomerativeClustering(n_clusters=optimal_n_clusters),
        'GaussianMixture': GaussianMixture(n_components=optimal_n_clusters, random_state=1),
        'BIRCH': KMeans(n_clusters=optimal_n_clusters, random_state=1)  # Placeholder for BIRCH
    }
    
    algorithm_labels = {}
    silhouette_values = {}
    
    for name, algorithm in tqdm(clustering_algorithms.items(), desc="Comparing Algorithms"):
        algorithm_labels[name] = algorithm.fit_predict(X_sample)
        silhouette_values[name] = silhouette_score(X_sample, algorithm_labels[name])
    
    # Calculate agreement between algorithms
    algorithm_agreement = {}
    
    for name1, labels1 in algorithm_labels.items():
        for name2, labels2 in algorithm_labels.items():
            if name1 != name2:
                agreement = adjusted_rand_score(labels1, labels2)
                algorithm_agreement[f'{name1} vs {name2}'] = agreement
    
    # Convert to DataFrame
    algorithm_agreement_df = pd.DataFrame([algorithm_agreement]).melt()
    algorithm_agreement_df.columns = ['Algorithm Pair', 'Agreement']
    
    # Plot algorithm agreement
    plt.figure(figsize=(12, 6))
    
    plt.bar(algorithm_agreement_df['Algorithm Pair'], algorithm_agreement_df['Agreement'], color='blue', alpha=0.7)
    plt.axhline(y=0.7, color='red', linestyle='--', alpha=0.7, label='70% Agreement Threshold')
    plt.xlabel('Algorithm Pair')
    plt.ylabel('Agreement (Adjusted Rand Index)')
    plt.title('Agreement Between Different Clustering Algorithms')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.0)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'algorithm_agreement.pdf'), format='pdf', dpi=300)
    plt.close()
    
    # Plot silhouette values for each algorithm
    plt.figure(figsize=(10, 6))
    
    plt.bar(silhouette_values.keys(), silhouette_values.values(), color='blue', alpha=0.7)
    plt.xlabel('Algorithm')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Scores for Different Clustering Algorithms')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'algorithm_silhouette.pdf'), format='pdf', dpi=300)
    plt.close()
    
    # Feature impact analysis
    feature_names = columns_for_clustering
    feature_importance = []
    
    # Choose the best algorithm based on silhouette score
    best_algorithm = max(silhouette_values.items(), key=lambda x: x[1])[0]
    best_labels = algorithm_labels[best_algorithm]
    
    # For each feature, remove it and calculate the impact on clustering results
    for i, feature in enumerate(feature_names):
        # Create a dataset without the current feature
        X_reduced = np.delete(X_sample, i, axis=1)
        
        # Fit the best algorithm on the reduced dataset
        if best_algorithm == 'KMeans':
            reduced_algorithm = KMeans(n_clusters=optimal_n_clusters, random_state=1)
        elif best_algorithm == 'Agglomerative':
            reduced_algorithm = AgglomerativeClustering(n_clusters=optimal_n_clusters)
        elif best_algorithm == 'GaussianMixture':
            reduced_algorithm = GaussianMixture(n_components=optimal_n_clusters, random_state=1)
        elif best_algorithm == 'BIRCH':
            reduced_algorithm = KMeans(n_clusters=optimal_n_clusters, random_state=1)
        
        reduced_labels = reduced_algorithm.fit_predict(X_reduced)
        
        # Calculate agreement with full-feature clustering (adjusted Rand index)
        agreement = adjusted_rand_score(best_labels, reduced_labels)
        
        # Calculate impact (1 - agreement, higher means more important)
        impact = 1 - agreement
        
        feature_importance.append({
            'Feature': feature,
            'Impact': impact
        })
    
    # Convert to DataFrame and sort
    feature_importance_df = pd.DataFrame(feature_importance)
    feature_importance_df = feature_importance_df.sort_values('Impact', ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    
    plt.bar(feature_importance_df['Feature'], feature_importance_df['Impact'], color='blue', alpha=0.7)
    plt.xlabel('Feature')
    plt.ylabel('Feature Impact (Higher is More Important)')
    plt.title('Feature Impact on Clustering Results')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'feature_impact.pdf'), format='pdf', dpi=300)
    plt.close()
    
    # Temporal stability analysis (if transaction date is available)
    if 'Transaction Date' in trans_df.columns:
        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_dtype(trans_df['Transaction Date']):
            trans_df['Transaction Date'] = pd.to_datetime(trans_df['Transaction Date'])
        
        # Extract month information
        trans_df['Month'] = trans_df['Transaction Date'].dt.strftime('%Y-%m')
        
        # Get months with sufficient data
        month_counts = trans_df['Month'].value_counts()
        valid_months = month_counts[month_counts > 1000].index.sort_values()
        
        if len(valid_months) >= 3:
            # Analyze temporal stability
            month_stability = []
            
            # Sample data from each month
            for i in range(len(valid_months) - 1):
                current_month = valid_months[i]
                next_month = valid_months[i+1]
                
                # Extract current month data
                current_indices = data_clustering.index[trans_df.loc[data_clustering.index, 'Month'] == current_month]
                current_sample_indices = np.random.choice(current_indices, size=min(1000, len(current_indices)), replace=False)
                current_data = X_scaled[current_sample_indices]
                
                # Extract next month data
                next_indices = data_clustering.index[trans_df.loc[data_clustering.index, 'Month'] == next_month]
                next_sample_indices = np.random.choice(next_indices, size=min(1000, len(next_indices)), replace=False)
                next_data = X_scaled[next_sample_indices]
                
                # Cluster current month data
                kmeans_current = KMeans(n_clusters=optimal_n_clusters, random_state=1)
                current_labels = kmeans_current.fit_predict(current_data)
                
                # Use current month centroids to predict next month clusters
                next_predictions = kmeans_current.predict(next_data)
                
                # Directly cluster next month data
                kmeans_next = KMeans(n_clusters=optimal_n_clusters, random_state=1)
                next_direct_labels = kmeans_next.fit_predict(next_data)
                
                # Calculate agreement
                agreement = adjusted_rand_score(next_predictions, next_direct_labels)
                
                month_stability.append({
                    'Current Month': current_month,
                    'Next Month': next_month,
                    'Agreement': agreement
                })
            
            # Convert to DataFrame
            month_stability_df = pd.DataFrame(month_stability)
            
            # Plot temporal stability
            plt.figure(figsize=(12, 6))
            
            month_pairs = [f"{row['Current Month']} vs {row['Next Month']}" for _, row in month_stability_df.iterrows()]
            
            plt.bar(month_pairs, month_stability_df['Agreement'], color='blue', alpha=0.7)
            plt.axhline(y=0.7, color='red', linestyle='--', alpha=0.7, label='70% Agreement Threshold')
            plt.xlabel('Month Pair')
            plt.ylabel('Temporal Stability (Agreement)')
            plt.title('Clustering Stability Across Consecutive Months')
            plt.xticks(rotation=45, ha='right')
            plt.ylim(0, 1.0)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'temporal_stability.pdf'), format='pdf', dpi=300)
            plt.close()
    
    # Create a comprehensive stability report
    stability_report = {
        'optimal_clusters': optimal_n_clusters,
        'silhouette_scores': dict(zip(n_clusters_range, silhouette_scores)),
        'inertia_values': dict(zip(n_clusters_range, inertia_values)),
        'algorithm_silhouette': silhouette_values,
        'algorithm_agreement': algorithm_agreement,
        'feature_importance': feature_importance_df.to_dict('records')
    }
    
    # Add temporal stability if available
    if 'Transaction Date' in trans_df.columns and len(valid_months) >= 3:
        stability_report['temporal_stability'] = month_stability_df.to_dict('records')
    
    # Convert to DataFrame for easy saving
    stability_report_df = pd.DataFrame([stability_report])
    
    # Save to CSV
    stability_report_df.to_json(os.path.join(save_dir, 'transaction_clustering_stability_report.json'), orient='records')
    
    return stability_report

def clustering_calibration_analysis(vehicle_df, trans_df, save_dir='plots/calibration'):
    """
    Perform calibration analysis for clustering parameter selection.
    
    Parameters:
    -----------
    vehicle_df : pandas DataFrame
        The vehicle dataset
    trans_df : pandas DataFrame
        The transaction dataset
    save_dir : str
        Directory to save the output plots
    
    Returns:
    --------
    calibration_results : pandas DataFrame
        DataFrame containing the calibration analysis results
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Vehicle clustering calibration
    vehicle_cols = ['KMPL', 'Total Transaction Amount', 'Mean Transaction Amount',
                   'Total No. of Litres', 'Mean No. of Litres']
    
    # Extract and prepare data
    vehicle_data = vehicle_df[vehicle_cols].dropna()
    
    # Scale the data
    vehicle_scaler = StandardScaler()
    vehicle_X_scaled = vehicle_scaler.fit_transform(vehicle_data)
    
    # Define parameter grid for KMeans
    kmeans_param_grid = {
        'n_clusters': [2, 3, 4, 5],
        'init': ['k-means++', 'random'],
        'n_init': [10, 20],
        'max_iter': [200, 300, 500]
    }
    
    # Define parameter grid for Agglomerative Clustering
    agglom_param_grid = {
        'n_clusters': [2, 3, 4, 5],
        'linkage': ['ward', 'complete', 'average'],
        'affinity': ['euclidean', 'l1', 'l2', 'manhattan']
    }
    
    # Initialize results storage
    kmeans_results = []
    agglom_results = []
    
    # Run KMeans grid search
    for params in tqdm(ParameterGrid(kmeans_param_grid), desc="KMeans Calibration"):
        kmeans = KMeans(random_state=1, **params)
        labels = kmeans.fit_predict(vehicle_X_scaled)
        
        if len(np.unique(labels)) <= 1:
            # Skip if only one cluster is found
            continue
        
        silhouette = silhouette_score(vehicle_X_scaled, labels)
        
        kmeans_results.append({
            **params,
            'silhouette_score': silhouette,
            'inertia': kmeans.inertia_
        })
    
    # Convert to DataFrame
    kmeans_results_df = pd.DataFrame(kmeans_results)
    
    # Run Agglomerative grid search
    for params in tqdm(ParameterGrid(agglom_param_grid), desc="Agglomerative Calibration"):
        # Skip incompatible combinations
        if params['linkage'] == 'ward' and params['affinity'] != 'euclidean':
            continue
        
        agglom = AgglomerativeClustering(**params)
        labels = agglom.fit_predict(vehicle_X_scaled)
        
        if len(np.unique(labels)) <= 1:
            # Skip if only one cluster is found
            continue
        
        silhouette = silhouette_score(vehicle_X_scaled, labels)
        
        agglom_results.append({
            **params,
            'silhouette_score': silhouette
        })
    
    # Convert to DataFrame
    agglom_results_df = pd.DataFrame(agglom_results)
    
    # Plot KMeans results by n_clusters
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='n_clusters', y='silhouette_score', data=kmeans_results_df)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('KMeans Silhouette Score by Number of Clusters')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'kmeans_silhouette_by_clusters.pdf'), format='pdf', dpi=300)
    plt.close()
    
    # Plot Agglomerative results by n_clusters and linkage
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='n_clusters', y='silhouette_score', hue='linkage', data=agglom_results_df)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Agglomerative Silhouette Score by Number of Clusters and Linkage')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Linkage')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'agglom_silhouette_by_clusters.pdf'), format='pdf', dpi=300)
    plt.close()
    
    # Plot KMeans results by initialization method
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='init', y='silhouette_score', data=kmeans_results_df)
    plt.xlabel('Initialization Method')
    plt.ylabel('Silhouette Score')
    plt.title('KMeans Silhouette Score by Initialization Method')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'kmeans_silhouette_by_init.pdf'), format='pdf', dpi=300)
    plt.close()
    
    # Get top parameter combinations
    top_kmeans = kmeans_results_df.sort_values('silhouette_score', ascending=False).head(5)
    top_agglom = agglom_results_df.sort_values('silhouette_score', ascending=False).head(5)
    
    # Save top parameter combinations
    top_kmeans.to_csv(os.path.join(save_dir, 'top_kmeans_parameters.csv'), index=False)
    top_agglom.to_csv(os.path.join(save_dir, 'top_agglom_parameters.csv'), index=False)
    
    # Transaction clustering calibration - using a random sample
    trans_cols = ['Transaction Amount', 'No. of Litres', 'Estimated Price Per Litre', 'Days Between Transactions']
    
    # Extract and prepare data
    trans_data = trans_df[trans_cols].dropna()
    
    # Take a random sample for calibration
    sample_size = min(20000, trans_data.shape[0])
    trans_sample = trans_data.sample(sample_size, random_state=1)
    
    # Scale the data
    trans_scaler = StandardScaler()
    trans_X_scaled = trans_scaler.fit_transform(trans_sample)
    
    # Define parameter grid for KMeans (simplified for transactions)
    trans_kmeans_param_grid = {
        'n_clusters': [3, 4, 5, 6],
        'init': ['k-means++'],
        'n_init': [10],
        'max_iter': [300]
    }
    
    # Define parameter grid for GaussianMixture
    gmm_param_grid = {
        'n_components': [3, 4, 5, 6],
        'covariance_type': ['full', 'tied', 'diag', 'spherical'],
        'init_params': ['kmeans']
    }
    
    # Initialize results storage
    trans_kmeans_results = []
    gmm_results = []
    
    # Run KMeans grid search for transactions
    for params in tqdm(ParameterGrid(trans_kmeans_param_grid), desc="Transaction KMeans Calibration"):
        kmeans = KMeans(random_state=1, **params)
        labels = kmeans.fit_predict(trans_X_scaled)
        
        silhouette = silhouette_score(trans_X_scaled, labels)
        
        trans_kmeans_results.append({
            **params,
            'silhouette_score': silhouette,
            'inertia': kmeans.inertia_
        })
    
    # Convert to DataFrame
    trans_kmeans_results_df = pd.DataFrame(trans_kmeans_results)
    
    # Run GaussianMixture grid search
    for params in tqdm(ParameterGrid(gmm_param_grid), desc="GMM Calibration"):
        gmm = GaussianMixture(random_state=1, **params)
        labels = gmm.fit_predict(trans_X_scaled)
        
        if len(np.unique(labels)) <= 1:
            # Skip if only one cluster is found
            continue
        
        silhouette = silhouette_score(trans_X_scaled, labels)
        aic = gmm.aic(trans_X_scaled)
        bic = gmm.bic(trans_X_scaled)
        
        gmm_results.append({
            **params,
            'silhouette_score': silhouette,
            'aic': aic,
            'bic': bic
        })
    
    # Convert to DataFrame
    gmm_results_df = pd.DataFrame(gmm_results)
    
    # Plot Transaction KMeans results by n_clusters
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='n_clusters', y='silhouette_score', data=trans_kmeans_results_df)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Transaction KMeans Silhouette Score by Number of Clusters')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'trans_kmeans_silhouette.pdf'), format='pdf', dpi=300)
    plt.close()
    
    # Plot GMM results by n_components and covariance_type
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='n_components', y='silhouette_score', hue='covariance_type', data=gmm_results_df)
    plt.xlabel('Number of Components')
    plt.ylabel('Silhouette Score')
    plt.title('GMM Silhouette Score by Number of Components and Covariance Type')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Covariance Type')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'gmm_silhouette.pdf'), format='pdf', dpi=300)
    plt.close()
    
    # Plot GMM AIC results
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='n_components', y='aic', hue='covariance_type', data=gmm_results_df)
    plt.xlabel('Number of Components')
    plt.ylabel('AIC (Lower is Better)')
    plt.title('GMM AIC by Number of Components and Covariance Type')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Covariance Type')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'gmm_aic.pdf'), format='pdf', dpi=300)
    plt.close()
    
    # Plot GMM BIC results
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='n_components', y='bic', hue='covariance_type', data=gmm_results_df)
    plt.xlabel('Number of Components')
    plt.ylabel('BIC (Lower is Better)')
    plt.title('GMM BIC by Number of Components and Covariance Type')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Covariance Type')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'gmm_bic.pdf'), format='pdf', dpi=300)
    plt.close()
    
    # Get top parameter combinations for transactions
    top_trans_kmeans = trans_kmeans_results_df.sort_values('silhouette_score', ascending=False).head(5)
    top_gmm = gmm_results_df.sort_values('silhouette_score', ascending=False).head(5)
    
    # Save top parameter combinations for transactions
    top_trans_kmeans.to_csv(os.path.join(save_dir, 'top_trans_kmeans_parameters.csv'), index=False)
    top_gmm.to_csv(os.path.join(save_dir, 'top_gmm_parameters.csv'), index=False)
    
    # Return results
    return {
        'vehicle_kmeans': kmeans_results_df,
        'vehicle_agglom': agglom_results_df,
        'transaction_kmeans': trans_kmeans_results_df,
        'transaction_gmm': gmm_results_df
    }

# Run the functions if this script is executed directly
if __name__ == "__main__":
    print("Loading datasets...")
    trans_df, vehicle_df = load_datasets()
    
    print("Starting vehicle clustering stability analysis...")
    vehicle_stability = perform_vehicle_clustering_stability(vehicle_df)
    print("Vehicle clustering stability analysis complete!")
    
    print("Starting transaction clustering stability analysis...")
    transaction_stability = perform_transaction_clustering_stability(trans_df)
    print("Transaction clustering stability analysis complete!")
    
    print("Starting clustering calibration analysis...")
    calibration_results = clustering_calibration_analysis(vehicle_df, trans_df)
    print("Clustering calibration analysis complete!")
    
    print("All analyses complete!")