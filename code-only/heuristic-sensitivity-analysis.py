#!/usr/bin/env python
# coding: utf-8

# # Heuristic Flags Sensitivity Analysis
# 
# This script extends the original heuristic flags implementation by conducting comprehensive 
# sensitivity analysis on the parameters used for flagging suspicious transactions.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.ticker import FixedLocator, FixedFormatter
from tqdm import tqdm
import joblib

# Set the aesthetic style of the plots
sns.set_style("whitegrid")
plt.rc('axes', titlesize=16)
plt.rc('axes', labelsize=15)
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
sns.set_palette('cividis')

# Read in the dataset
data = pd.read_csv(os.path.join("data", "Final for clustering.csv"))
data_orig = data.copy()  # Keep an original copy for comparison

def perform_flag_sensitivity(df, multiplier_range, days_range, price_diff_range, save_dir='plots/sensitivity'):
    """
    Perform sensitivity analysis on all three heuristic flags.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The transaction dataset
    multiplier_range : list
        List of multipliers to test for the Transaction_Amount_Flag
    days_range : list
        List of day thresholds to test for the Transaction_Frequency_Flag
    price_diff_range : list
        List of price difference thresholds to test for the Fuel_Price_Flag
    save_dir : str
        Directory to save the output plots
    
    Returns:
    --------
    results_df : pandas DataFrame
        DataFrame containing the sensitivity analysis results
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize results dataframe
    results = []
    
    # Status tracking with tqdm
    total_iterations = len(multiplier_range) * len(days_range) * len(price_diff_range)
    pbar = tqdm(total=total_iterations, desc="Sensitivity Analysis Progress")
    
    for multiplier in multiplier_range:
        for days_threshold in days_range:
            for price_diff in price_diff_range:
                # Make a copy of the original data
                data = df.copy()
                
                # 1. Transaction Amount Flag with current multiplier
                data['Average_Category_Amount'] = data.groupby(['RATE CARD CATEGORY', 'District', 'Month Name'])['Transaction Amount'].transform('mean')
                data['Transaction_Amount_Flag'] = data['Transaction Amount'] > data['Average_Category_Amount'] * multiplier
                
                # 2. Transaction Frequency Flag with current days_threshold
                data['Transaction Date'] = pd.to_datetime(data['Transaction Date'])
                data.sort_values(by=['REG_NUM', 'Transaction Date'], inplace=True)
                data['Days_Between_Transactions'] = data.groupby('REG_NUM')['Transaction Date'].diff().dt.days
                data['Transaction_Frequency_Flag'] = (data['Days_Between_Transactions'] < days_threshold) & (data['Transaction Amount'] > data['Average_Category_Amount'])
                
                # 3. Fuel Price Flag with current price_diff
                # Function to calculate if the difference exceeds the threshold for each transaction
                diesel_actual = [22.75, 23.34, 23.43]  # Actual diesel price
                gov_price = 20.64  # Government price
                mean_diesel = sum(diesel_actual) / 3  # Mean diesel price
                diff = mean_diesel - gov_price  # Difference between mean diesel price and government price
                
                # Create a new column called Coastal Diesel Adjusted for the difference
                data['Coastal Diesel Adjusted'] = data['Coastal Diesel'] + diff
                
                # Create a new column called price difference
                data['Price Difference'] = data.apply(lambda row: abs(row['Coastal Diesel Adjusted'] - row['Estimated Price Per Litre']) 
                                                   if row['Fuel Type'] == 'Diesel' 
                                                   else abs(row['Coastal Petrol'] - row['Estimated Price Per Litre']), axis=1)
                
                # Create a Fuel Price Flag column that flags transactions where the price difference is greater than the threshold
                data['Fuel_Price_Flag'] = data['Price Difference'] > price_diff
                
                # Create a new variable called number of flags that counts the number of flags for each transaction as an integer
                data['Number_of_Flags'] = data['Transaction_Amount_Flag'].astype(int) + data['Transaction_Frequency_Flag'].astype(int) + data['Fuel_Price_Flag'].astype(int)
                
                # Calculate flag distribution
                flag_dist = data['Number_of_Flags'].value_counts(normalize=True).sort_index()
                
                # Calculate metrics by department and district
                dept_flags = data.groupby('DEPARTMENT')['Number_of_Flags'].mean()
                district_flags = data.groupby('District')['Number_of_Flags'].mean()
                
                # Calculate financial impact
                avg_flagged_amount = data[data['Number_of_Flags'] >= 2]['Transaction Amount'].mean()
                total_flagged_amount = data[data['Number_of_Flags'] >= 2]['Transaction Amount'].sum()
                
                # Store results
                results.append({
                    'Multiplier': multiplier,
                    'Days_Threshold': days_threshold,
                    'Price_Diff': price_diff,
                    'Pct_No_Flags': flag_dist.get(0, 0) * 100,
                    'Pct_One_Flag': flag_dist.get(1, 0) * 100,
                    'Pct_Multi_Flags': (flag_dist.get(2, 0) + flag_dist.get(3, 0)) * 100,
                    'Num_Multi_Flags': data[data['Number_of_Flags'] >= 2].shape[0],
                    'Avg_Flagged_Amount': avg_flagged_amount,
                    'Total_Flagged_Amount': total_flagged_amount,
                    'Highest_Dept_Flags': dept_flags.idxmax(),
                    'Highest_Dept_Flag_Rate': dept_flags.max(),
                    'Highest_District_Flags': district_flags.idxmax(),
                    'Highest_District_Flag_Rate': district_flags.max()
                })
                
                pbar.update(1)
    
    pbar.close()
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results to CSV
    results_df.to_csv(os.path.join(save_dir, 'sensitivity_analysis_results.csv'), index=False)
    
    # Generate plots
    # 1. Heatmap of multi-flag percentage by multiplier and days_threshold
    # (averaging over price_diff)
    pivot_data = results_df.groupby(['Multiplier', 'Days_Threshold'])['Pct_Multi_Flags'].mean().reset_index()
    pivot_data = pivot_data.pivot(index='Multiplier', columns='Days_Threshold', values='Pct_Multi_Flags')
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_data, annot=True, fmt=".2f", cmap="cividis", 
                linewidths=.5, cbar_kws={'label': 'Percent with Multiple Flags'})
    plt.title('Sensitivity of Multiple Flag Percentage to Parameter Changes')
    plt.xlabel('Days Between Transactions Threshold')
    plt.ylabel('Transaction Amount Multiplier')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'multi_flag_sensitivity_heatmap.pdf'), format='pdf', dpi=300)
    plt.close()
    
    # 2. Line plots showing the effect of each parameter while holding others constant
    # Effect of multiplier
    default_days = 2
    default_price = 1.0
    
    multiplier_effect = results_df[(results_df['Days_Threshold'] == default_days) & 
                                  (results_df['Price_Diff'] == default_price)]
    
    plt.figure(figsize=(10, 6))
    plt.plot(multiplier_effect['Multiplier'], multiplier_effect['Pct_Multi_Flags'], 
             marker='o', linewidth=2)
    plt.axvline(x=1.5, color='red', linestyle='--', alpha=0.7, label='Current Setting (1.5)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Transaction Amount Multiplier')
    plt.ylabel('Percentage with Multiple Flags')
    plt.title('Effect of Transaction Amount Multiplier on Flag Rate')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'multiplier_effect.pdf'), format='pdf', dpi=300)
    plt.close()
    
    # Effect of days threshold
    default_multiplier = 1.5
    
    days_effect = results_df[(results_df['Multiplier'] == default_multiplier) & 
                            (results_df['Price_Diff'] == default_price)]
    
    plt.figure(figsize=(10, 6))
    plt.plot(days_effect['Days_Threshold'], days_effect['Pct_Multi_Flags'], 
             marker='o', linewidth=2)
    plt.axvline(x=2, color='red', linestyle='--', alpha=0.7, label='Current Setting (2 days)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Days Between Transactions Threshold')
    plt.ylabel('Percentage with Multiple Flags')
    plt.title('Effect of Days Threshold on Flag Rate')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'days_effect.pdf'), format='pdf', dpi=300)
    plt.close()
    
    # Effect of price difference
    price_effect = results_df[(results_df['Multiplier'] == default_multiplier) & 
                             (results_df['Days_Threshold'] == default_days)]
    
    plt.figure(figsize=(10, 6))
    plt.plot(price_effect['Price_Diff'], price_effect['Pct_Multi_Flags'], 
             marker='o', linewidth=2)
    plt.axvline(x=1.0, color='red', linestyle='--', alpha=0.7, label='Current Setting (R1)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Price Difference Threshold (R)')
    plt.ylabel('Percentage with Multiple Flags')
    plt.title('Effect of Price Difference Threshold on Flag Rate')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'price_effect.pdf'), format='pdf', dpi=300)
    plt.close()
    
    # 3. Bar chart showing the financial impact at different parameter combinations
    # Top 5 parameter combinations by total flagged amount
    top_financial = results_df.sort_values('Total_Flagged_Amount', ascending=False).head(5)
    
    # Create labels for the parameter combinations
    top_financial['Param_Combo'] = top_financial.apply(
        lambda row: f"M:{row['Multiplier']}, D:{row['Days_Threshold']}, P:R{row['Price_Diff']}", axis=1)
    
    plt.figure(figsize=(12, 6))
    plt.bar(top_financial['Param_Combo'], top_financial['Total_Flagged_Amount'] / 1000000)
    plt.axhline(y=results_df[(results_df['Multiplier'] == 1.5) & 
                             (results_df['Days_Threshold'] == 2) & 
                             (results_df['Price_Diff'] == 1.0)]['Total_Flagged_Amount'].values[0] / 1000000, 
                color='red', linestyle='--', alpha=0.7, label='Current Settings')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Parameter Combination')
    plt.ylabel('Total Flagged Amount (Millions R)')
    plt.title('Financial Impact of Top 5 Parameter Combinations')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'financial_impact.pdf'), format='pdf', dpi=300)
    plt.close()
    
    # 4. Create a table for the paper
    # Select key parameter combinations
    key_params = [
        (1.3, 2, 1.0),  # More sensitive multiplier
        (1.5, 2, 1.0),  # Current settings
        (1.7, 2, 1.0),  # Less sensitive multiplier
        (1.5, 1, 1.0),  # More sensitive days
        (1.5, 3, 1.0),  # Less sensitive days
        (1.5, 2, 0.8),  # More sensitive price
        (1.5, 2, 1.2)   # Less sensitive price
    ]
    
    key_results = results_df.loc[
        results_df.apply(lambda row: (row['Multiplier'], row['Days_Threshold'], row['Price_Diff']) in key_params, axis=1)
    ].sort_values(by=['Multiplier', 'Days_Threshold', 'Price_Diff'])
    
    # Add a description column
    descriptions = [
        'More sensitive amount threshold (1.3×)',
        'Current settings (1.5×, 2 days, R1)',
        'Less sensitive amount threshold (1.7×)',
        'More sensitive frequency threshold (1 day)',
        'Less sensitive frequency threshold (3 days)',
        'More sensitive price threshold (R0.8)',
        'Less sensitive price threshold (R1.2)'
    ]
    
    key_results['Description'] = descriptions
    
    # Calculate percentage changes relative to current settings
    current_settings = key_results.loc[1]  # Assuming current settings are the second row
    
    key_results['Pct_Change_Multi_Flags'] = ((key_results['Pct_Multi_Flags'] - current_settings['Pct_Multi_Flags']) / 
                                           current_settings['Pct_Multi_Flags'] * 100)
    
    key_results['Pct_Change_Total_Amount'] = ((key_results['Total_Flagged_Amount'] - current_settings['Total_Flagged_Amount']) / 
                                            current_settings['Total_Flagged_Amount'] * 100)
    
    # Reorder and select columns for the table
    table_cols = ['Description', 'Pct_No_Flags', 'Pct_One_Flag', 'Pct_Multi_Flags', 
                 'Num_Multi_Flags', 'Total_Flagged_Amount', 'Pct_Change_Multi_Flags', 'Pct_Change_Total_Amount']
    
    key_table = key_results[table_cols].copy()
    
    # Format the columns for display
    key_table['Pct_No_Flags'] = key_table['Pct_No_Flags'].round(2)
    key_table['Pct_One_Flag'] = key_table['Pct_One_Flag'].round(2)
    key_table['Pct_Multi_Flags'] = key_table['Pct_Multi_Flags'].round(2)
    key_table['Total_Flagged_Amount'] = key_table['Total_Flagged_Amount'].apply(lambda x: f"R{x:,.0f}")
    key_table['Pct_Change_Multi_Flags'] = key_table['Pct_Change_Multi_Flags'].apply(lambda x: f"{x:+.2f}%")
    key_table['Pct_Change_Total_Amount'] = key_table['Pct_Change_Total_Amount'].apply(lambda x: f"{x:+.2f}%")
    
    # Save to CSV
    key_table.to_csv(os.path.join(save_dir, 'key_parameter_comparison.csv'), index=False)
    
    return results_df

def apply_optimal_parameters(df, multiplier, days_threshold, price_diff, output_path):
    """
    Apply the optimal parameters identified through sensitivity analysis
    and save the resulting dataset.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The transaction dataset
    multiplier : float
        Multiplier for the Transaction_Amount_Flag
    days_threshold : int
        Days threshold for the Transaction_Frequency_Flag
    price_diff : float
        Price difference threshold for the Fuel_Price_Flag
    output_path : str
        Path to save the output dataset
    
    Returns:
    --------
    data : pandas DataFrame
        DataFrame with optimized flags applied
    """
    # Make a copy of the original data
    data = df.copy()
    
    # 1. Transaction Amount Flag with optimal multiplier
    data['Average_Category_Amount'] = data.groupby(['RATE CARD CATEGORY', 'District', 'Month Name'])['Transaction Amount'].transform('mean')
    data['Transaction_Amount_Flag'] = data['Transaction Amount'] > data['Average_Category_Amount'] * multiplier
    
    # 2. Transaction Frequency Flag with optimal days_threshold
    data['Transaction Date'] = pd.to_datetime(data['Transaction Date'])
    data.sort_values(by=['REG_NUM', 'Transaction Date'], inplace=True)
    data['Days_Between_Transactions'] = data.groupby('REG_NUM')['Transaction Date'].diff().dt.days
    data['Transaction_Frequency_Flag'] = (data['Days_Between_Transactions'] < days_threshold) & (data['Transaction Amount'] > data['Average_Category_Amount'])
    
    # 3. Fuel Price Flag with optimal price_diff
    diesel_actual = [22.75, 23.34, 23.43]  # Actual diesel price
    gov_price = 20.64  # Government price
    mean_diesel = sum(diesel_actual) / 3  # Mean diesel price
    diff = mean_diesel - gov_price  # Difference between mean diesel price and government price
    
    # Create a new column called Coastal Diesel Adjusted for the difference
    data['Coastal Diesel Adjusted'] = data['Coastal Diesel'] + diff
    
    # Create a new column called price difference
    data['Price Difference'] = data.apply(lambda row: abs(row['Coastal Diesel Adjusted'] - row['Estimated Price Per Litre']) 
                                       if row['Fuel Type'] == 'Diesel' 
                                       else abs(row['Coastal Petrol'] - row['Estimated Price Per Litre']), axis=1)
    
    # Create a Fuel Price Flag column that flags transactions where the price difference is greater than the threshold
    data['Fuel_Price_Flag'] = data['Price Difference'] > price_diff
    
    # Create a new variable called number of flags that counts the number of flags for each transaction as an integer
    data['Number_of_Flags'] = data['Transaction_Amount_Flag'].astype(int) + data['Transaction_Frequency_Flag'].astype(int) + data['Fuel_Price_Flag'].astype(int)
    
    # Convert Number_of_Flags to a categorical variable
    data['Number_of_Flags'] = data['Number_of_Flags'].astype('category')
    
    # Save the data to a new file
    data.to_csv(output_path, index=False)
    
    return data

def cross_validate_flags(df, output_dir='../results/flag_validation'):
    """
    Perform cross-validation of the flag approach by splitting the data
    temporally and testing flag consistency across time periods.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The transaction dataset
    output_dir : str
        Directory to save the output files
    
    Returns:
    --------
    validation_results : pandas DataFrame
        DataFrame containing the validation results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to datetime if needed
    if not pd.api.types.is_datetime64_dtype(df['Transaction Date']):
        df['Transaction Date'] = pd.to_datetime(df['Transaction Date'])
    
    # Sort by transaction date
    df = df.sort_values('Transaction Date')
    
    # Create a month column
    df['Month'] = df['Transaction Date'].dt.to_period('M')
    
    # Get unique months
    months = df['Month'].unique()
    
    # Ensure at least 3 months of data
    if len(months) < 3:
        raise ValueError("Need at least 3 months of data for temporal cross-validation")
    
    # Initialize results storage
    results = []
    
    # Use a 3-month sliding window for validation
    for i in range(len(months) - 2):
        # Training period: month i
        train_period = months[i]
        # Validation period: month i+1
        val_period = months[i+1]
        # Test period: month i+2
        test_period = months[i+2]
        
        # Split data
        train_data = df[df['Month'] == train_period].copy()
        val_data = df[df['Month'] == val_period].copy()
        test_data = df[df['Month'] == test_period].copy()
        
        # Calculate average metrics for training period
        train_data['Average_Category_Amount'] = train_data.groupby(['RATE CARD CATEGORY', 'District'])['Transaction Amount'].transform('mean')
        
        # Loop through parameter combinations
        for multiplier in [1.3, 1.5, 1.7]:
            for days_threshold in [1, 2, 3]:
                for price_diff in [0.8, 1.0, 1.2]:
                    # Apply flags to validation set
                    val_data['Transaction_Amount_Flag'] = val_data['Transaction Amount'] > val_data['Average_Category_Amount'] * multiplier
                    
                    # Sort for calculating days between transactions
                    val_data = val_data.sort_values(by=['REG_NUM', 'Transaction Date'])
                    val_data['Days_Between_Transactions'] = val_data.groupby('REG_NUM')['Transaction Date'].diff().dt.days
                    val_data['Transaction_Frequency_Flag'] = (val_data['Days_Between_Transactions'] < days_threshold) & (val_data['Transaction Amount'] > val_data['Average_Category_Amount'])
                    
                    # Fuel price flag
                    diesel_actual = [22.75, 23.34, 23.43]
                    gov_price = 20.64
                    mean_diesel = sum(diesel_actual) / 3
                    diff = mean_diesel - gov_price
                    
                    val_data['Coastal Diesel Adjusted'] = val_data['Coastal Diesel'] + diff
                    val_data['Price Difference'] = val_data.apply(lambda row: abs(row['Coastal Diesel Adjusted'] - row['Estimated Price Per Litre']) 
                                                                if row['Fuel Type'] == 'Diesel' 
                                                                else abs(row['Coastal Petrol'] - row['Estimated Price Per Litre']), axis=1)
                    val_data['Fuel_Price_Flag'] = val_data['Price Difference'] > price_diff
                    
                    # Count flags
                    val_data['Number_of_Flags'] = val_data['Transaction_Amount_Flag'].astype(int) + val_data['Transaction_Frequency_Flag'].astype(int) + val_data['Fuel_Price_Flag'].astype(int)
                    
                    # Update the parameters with validation results
                    # This is like a form of Bayesian updating
                    val_avg_category = val_data.groupby(['RATE CARD CATEGORY', 'District'])['Transaction Amount'].mean()
                    
                    # Combine training and validation results
                    combined_avg = (train_data.groupby(['RATE CARD CATEGORY', 'District'])['Transaction Amount'].mean() * 0.7 + 
                                  val_avg_category * 0.3)
                    
                    # Create a mapping for updating test data
                    avg_mapping = combined_avg.reset_index()
                    avg_mapping_dict = avg_mapping.set_index(['RATE CARD CATEGORY', 'District'])['Transaction Amount'].to_dict()
                    
                    # Apply to test data
                    test_data['Average_Category_Amount'] = test_data.apply(
                        lambda row: avg_mapping_dict.get((row['RATE CARD CATEGORY'], row['District']), 
                                                       row['Transaction Amount']), 
                        axis=1)
                    
                    # Apply flags to test set
                    test_data['Transaction_Amount_Flag'] = test_data['Transaction Amount'] > test_data['Average_Category_Amount'] * multiplier
                    
                    # Sort for calculating days between transactions
                    test_data = test_data.sort_values(by=['REG_NUM', 'Transaction Date'])
                    test_data['Days_Between_Transactions'] = test_data.groupby('REG_NUM')['Transaction Date'].diff().dt.days
                    test_data['Transaction_Frequency_Flag'] = (test_data['Days_Between_Transactions'] < days_threshold) & (test_data['Transaction Amount'] > test_data['Average_Category_Amount'])
                    
                    # Fuel price flag
                    test_data['Coastal Diesel Adjusted'] = test_data['Coastal Diesel'] + diff
                    test_data['Price Difference'] = test_data.apply(lambda row: abs(row['Coastal Diesel Adjusted'] - row['Estimated Price Per Litre']) 
                                                                 if row['Fuel Type'] == 'Diesel' 
                                                                 else abs(row['Coastal Petrol'] - row['Estimated Price Per Litre']), axis=1)
                    test_data['Fuel_Price_Flag'] = test_data['Price Difference'] > price_diff
                    
                    # Count flags
                    test_data['Number_of_Flags'] = test_data['Transaction_Amount_Flag'].astype(int) + test_data['Transaction_Frequency_Flag'].astype(int) + test_data['Fuel_Price_Flag'].astype(int)
                    
                    # Calculate validation and test metrics
                    val_flag_dist = val_data['Number_of_Flags'].value_counts(normalize=True).sort_index()
                    test_flag_dist = test_data['Number_of_Flags'].value_counts(normalize=True).sort_index()
                    
                    # Calculate consistency metrics
                    consistency_score = 1 - abs(val_flag_dist.get(2, 0) + val_flag_dist.get(3, 0) - 
                                             test_flag_dist.get(2, 0) - test_flag_dist.get(3, 0))
                    
                    # Calculate stability of flagged vehicles
                    val_flagged_vehicles = set(val_data[val_data['Number_of_Flags'] >= 2]['REG_NUM'].unique())
                    test_flagged_vehicles = set(test_data[test_data['Number_of_Flags'] >= 2]['REG_NUM'].unique())
                    
                    if len(val_flagged_vehicles) > 0 and len(test_flagged_vehicles) > 0:
                        vehicle_overlap = len(val_flagged_vehicles.intersection(test_flagged_vehicles)) / len(val_flagged_vehicles.union(test_flagged_vehicles))
                    else:
                        vehicle_overlap = 0
                    
                    # Store results
                    results.append({
                        'Train_Period': train_period,
                        'Val_Period': val_period,
                        'Test_Period': test_period,
                        'Multiplier': multiplier,
                        'Days_Threshold': days_threshold,
                        'Price_Diff': price_diff,
                        'Val_Pct_Multi_Flags': (val_flag_dist.get(2, 0) + val_flag_dist.get(3, 0)) * 100,
                        'Test_Pct_Multi_Flags': (test_flag_dist.get(2, 0) + test_flag_dist.get(3, 0)) * 100,
                        'Consistency_Score': consistency_score * 100,
                        'Vehicle_Overlap': vehicle_overlap * 100
                    })
    
    # Convert results to DataFrame
    validation_results = pd.DataFrame(results)
    
    # Save results
    validation_results.to_csv(os.path.join(output_dir, 'temporal_validation_results.csv'), index=False)
    
    # Create a summary plot of consistency scores by parameter combination
    summary = validation_results.groupby(['Multiplier', 'Days_Threshold', 'Price_Diff'])[
        ['Consistency_Score', 'Vehicle_Overlap']].mean().reset_index()
    
    # Plot consistency scores
    plt.figure(figsize=(12, 8))
    
    # Create parameter combination labels
    summary['Param_Combo'] = summary.apply(
        lambda row: f"M:{row['Multiplier']}, D:{row['Days_Threshold']}, P:R{row['Price_Diff']}", axis=1)
    
    # Sort by consistency score
    summary = summary.sort_values('Consistency_Score', ascending=False)
    
    # Plot
    plt.bar(summary['Param_Combo'], summary['Consistency_Score'], alpha=0.7)
    plt.axhline(y=90, color='red', linestyle='--', alpha=0.7, label='90% Consistency Target')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Parameter Combination')
    plt.ylabel('Average Consistency Score (%)')
    plt.title('Temporal Consistency of Flag Rate by Parameter Combination')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'flag_consistency_by_parameters.pdf'), format='pdf', dpi=300)
    plt.close()
    
    # Create a scatter plot of consistency vs. vehicle overlap
    plt.figure(figsize=(10, 8))
    plt.scatter(summary['Consistency_Score'], summary['Vehicle_Overlap'], alpha=0.7)
    
    # Add parameter labels to points
    for i, row in summary.iterrows():
        plt.annotate(f"M:{row['Multiplier']}, D:{row['Days_Threshold']}, P:{row['Price_Diff']}", 
                   (row['Consistency_Score'], row['Vehicle_Overlap']),
                   textcoords="offset points", 
                   xytext=(0,10), 
                   ha='center',
                   fontsize=8)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Consistency Score (%)')
    plt.ylabel('Vehicle Overlap Between Periods (%)')
    plt.title('Trade-off Between Flag Rate Consistency and Vehicle Flagging Stability')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'consistency_vs_overlap.pdf'), format='pdf', dpi=300)
    plt.close()
    
    return validation_results

# Run the functions if this script is executed directly
if __name__ == "__main__":
    # Define parameter ranges for sensitivity analysis
    multiplier_range = [1.3, 1.4, 1.5, 1.6, 1.7]
    days_range = [1, 2, 3, 4]
    price_diff_range = [0.8, 0.9, 1.0, 1.1, 1.2]
    
    # Perform sensitivity analysis
    print("Starting sensitivity analysis...")
    sensitivity_results = perform_flag_sensitivity(data_orig, multiplier_range, days_range, price_diff_range)
    print("Sensitivity analysis complete!")
    
    # Perform temporal cross-validation
    print("Starting temporal cross-validation...")
    validation_results = cross_validate_flags(data_orig)
    print("Temporal cross-validation complete!")
    
    # Apply the optimal parameters identified (adjust these based on results)
    # For now, we'll use the original parameters as a placeholder
    optimal_multiplier = 1.5
    optimal_days = 2
    optimal_price_diff = 1.0
    
    print(f"Applying optimal parameters: Multiplier={optimal_multiplier}, Days={optimal_days}, Price Diff={optimal_price_diff}")
    optimized_data = apply_optimal_parameters(
        data_orig, 
        optimal_multiplier, 
        optimal_days, 
        optimal_price_diff, 
        '../data/Final Transactions With Optimized Flags.csv'
    )
    print("Optimal parameters applied and dataset saved!")
    
    print("All analyses complete!")